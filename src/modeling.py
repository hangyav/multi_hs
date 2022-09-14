from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import SequenceClassifierOutput

class MultiTaskModelWrapper:

    def __init__(self, model):
        self.model = model
        self._old_forward = model.forward
        model.forward = self.forward

    def _pre_forward(self, *args, **kwargs):
        kwargs.pop('task_name')

    def _post_forward(self, output, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        self._pre_forward(*args, **kwargs)
        output = self._old_forward(*args, **kwargs)
        self._post_forward(output, *args, **kwargs)

        return output


class HeadSelectionWrapper(MultiTaskModelWrapper):

    def __init__(self, model, head_dict):
        super().__init__(model)
        self.head_dict = head_dict

    def _pre_forward(self, *args, **kwargs):
        task_name = kwargs.pop('task_name')
        head = self.head_dict[task_name]

        if self.model.active_head != head:
            self.model.active_head = head


class AdapterSelectionWrapper(MultiTaskModelWrapper):

    def __init__(self, model, adapter_dict, active_adapter_dict=None,
                 freeze_model_core=True):
        super().__init__(model)
        self.adapter_dict = adapter_dict
        self.active_adapter_dict = (
            active_adapter_dict
            if active_adapter_dict else
            adapter_dict
        )
        self.freeze_model_core = freeze_model_core

    def _pre_forward(self, *args, **kwargs):
        task_name = kwargs.pop('task_name')
        adapter = self.adapter_dict[task_name]
        active_adapter = self.active_adapter_dict[task_name]

        self.model.train_adapter(adapter)
        self.model.set_active_adapters(active_adapter)
        if not self.freeze_model_core:
            # let's only do this if we need to unfreeze
            self.model.freeze_model(self.freeze_model_core)


class PromptSelectionWrapper(MultiTaskModelWrapper):

    def __init__(self, model, prompt_models_dict):
        super().__init__(model)
        self.prompt_models_dict = prompt_models_dict
        self.loss_fct = CrossEntropyLoss()

    def forward(self, *args, return_dict=None, **kwargs):
        assert len(args) == 0
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # FIXME this is not too nice but needed to avoid infinite loop
        if 'task_name' not in kwargs:
            # this method is called by the prompt_model
            res = self._old_forward(*args, **kwargs)
            return res

        task_name = kwargs.pop('task_name')
        prompt_model = self.prompt_models_dict[task_name]

        logits = prompt_model.forward(kwargs)
        loss = self.loss_fct(logits, kwargs['label'])

        if not return_dict:
            return (loss, logits, None)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
