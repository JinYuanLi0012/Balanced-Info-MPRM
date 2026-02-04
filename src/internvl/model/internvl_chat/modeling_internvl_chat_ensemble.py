import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_internvl_chat import InternVLChatModel


class InternVLChatEnsemblePRMModel(InternVLChatModel):
    def __init__(
        self,
        config,
        vision_model=None,
        language_model=None,
        use_flash_attn=True,
        prm_num_heads: int = 4,
        prm_div_lambda: float = 1e-3,
        prm_head_bias: bool = True,
    ):
        super().__init__(
            config=config,
            vision_model=vision_model,
            language_model=language_model,
            use_flash_attn=use_flash_attn,
        )
        self.prm_num_heads = int(prm_num_heads)
        self.prm_div_lambda = float(prm_div_lambda)

        hidden_size = self.language_model.config.hidden_size
        self.prm_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1, bias=prm_head_bias) for _ in range(self.prm_num_heads)]
        )

        # 保存每个头的初始化参数副本，用于 L2 正则；注册为 buffer 以随设备移动、参与保存
        for i, head in enumerate(self.prm_heads):
            for name, param in head.named_parameters():
                buf_name = f"_prm_head{i}__{name.replace('.', '__')}_init"
                self.register_buffer(buf_name, param.detach().clone(), persistent=True)

    def _l2_to_init(self, head_idx: int) -> torch.Tensor:
        total = 0.0
        head = self.prm_heads[head_idx]
        for name, param in head.named_parameters():
            buf_name = f"_prm_head{head_idx}__{name.replace('.', '__')}_init"
            init_param = getattr(self, buf_name)
            total = total + (param.float() - init_param.to(param.device).float()).pow(2).sum()
        return total

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # 与父类保持一致的视觉特征注入流程
        input_ids_flat = input_ids.reshape(B * N)
        selected = input_ids_flat == self.img_context_token_id
        ignore_flag = False
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        need_hidden = True if (labels is not None and loss_weight is None) else bool(output_hidden_states)
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        logits = outputs.logits

        # 1) 若是 packed 的 token 级损失（loss_weight 非空），走父类的 token 损失路径
        loss = None
        if labels is not None and loss_weight is not None:
            # 直接复用父类逻辑（简化：这里不重复实现，调用父类计算）
            # 为避免破坏父类，我们在本类中只在 ensemble 分支生效；pack 分支建议继续使用原训练脚本
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            shift_weights = shift_weights.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            shift_weights_sum = shift_weights.sum()
            loss = (loss * shift_weights).sum() / (shift_weights_sum + 1e-8)
            if ignore_flag:
                loss = loss * 0.0

        # 2) 否则走 Ensemble PRM 分支（在 <prm> 处做多头 BCE + L2 正则）
        elif labels is not None:
            if not need_hidden or outputs.hidden_states is None:
                raise RuntimeError("hidden_states is required for ensemble PRM but not provided.")

            last_hidden = outputs.hidden_states[-1]  # [B, N, H]
            last_hidden = last_hidden.reshape(B * N, last_hidden.size(-1))
            placeholder_mask = input_ids_flat == self.prm_token_id
            if placeholder_mask.sum() == 0:
                # 没有 <prm> 的样本，返回零损防止梯度为 None
                loss = logits.sum() * 0.0
            else:
                feat_prm = last_hidden[placeholder_mask]          # [M, H]
                target = labels.reshape(-1)[placeholder_mask]     # [M], 软标签 [0,1]

                bce = nn.BCEWithLogitsLoss(reduction='mean')
                losses = []
                for i, head in enumerate(self.prm_heads):
                    head_logits = head(feat_prm).squeeze(-1)      # [M]
                    cls_loss = bce(head_logits, target.to(head_logits.dtype))
                    reg_loss = self._l2_to_init(i) * self.prm_div_lambda
                    losses.append(cls_loss + reg_loss)
                loss = sum(losses) / float(self.prm_num_heads)
                if ignore_flag:
                    loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def prm_ensemble(
        self,
        tokenizer,
        pixel_values,
        question,
        num_patches_list=None,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        PRM_TOKEN='<prm>',
        verbose=False,
    ):
        # 参考父类 prm()，但返回多头概率
        from internvl.conversation import get_conv_template
        prm_token_id = tokenizer.convert_tokens_to_ids(PRM_TOKEN)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = ([pixel_values.shape[0]] if pixel_values is not None else [])
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        template = get_conv_template(self.template)
        template.append_message(template.roles[0], '')
        template.append_message(template.roles[1], question)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=torch.tensor([1] * pixel_values.shape[0], dtype=torch.long, device=device),
            return_dict=True,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]      # [1, N, H]
        input_ids_flat = input_ids.view(-1)
        mask = input_ids_flat == prm_token_id
        if mask.sum() == 0:
            return torch.empty(self.prm_num_heads, 0)

        feat = last_hidden.view(-1, last_hidden.size(-1))[mask]   # [M, H]
        probs = []
        for head in self.prm_heads:
            logits = head(feat).squeeze(-1)
            probs.append(torch.sigmoid(logits))
        return torch.stack(probs, dim=0)   # [num_heads, M]

    @torch.no_grad()
    def batch_prm_ensemble(
        self,
        tokenizer,
        pixel_values,
        questions,
        num_patches_list=None,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        PRM_TOKEN='<prm>',
        verbose=False,
    ):
        # 与 batch_prm 类似，返回一个 list，每个元素为 [num_heads, M_i] 的概率张量
        from internvl.conversation import get_conv_template
        prm_token_id = tokenizer.convert_tokens_to_ids(PRM_TOKEN)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            q = questions[idx]
            if pixel_values is not None and '<image>' not in q:
                q = '<image>\n' + q
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], '')
            template.append_message(template.roles[1], q)
            query = template.get_prompt()
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=torch.tensor([1] * pixel_values.shape[0], dtype=torch.long, device=device),
            return_dict=True,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]      # [B, N, H]
        B, N, H = last_hidden.shape
        results = []
        for b in range(B):
            ids_b = input_ids[b]
            mask_b = ids_b == prm_token_id
            if mask_b.sum() == 0:
                results.append(torch.empty(self.prm_num_heads, 0, device=last_hidden.device))
                continue
            feat_b = last_hidden[b][mask_b]         # [M_b, H]
            probs = []
            for head in self.prm_heads:
                logits = head(feat_b).squeeze(-1)
                probs.append(torch.sigmoid(logits))
            results.append(torch.stack(probs, dim=0))   # [num_heads, M_b]
        return results