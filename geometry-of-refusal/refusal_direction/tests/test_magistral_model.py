"""Magistral factory and integration smoke tests (no model weights downloaded)."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from huggingface_hub import hf_hub_download
from transformers import MistralCommonTokenizer, MistralConfig, MistralForCausalLM

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.model_utils.magistral_model import MagistralModel, tokenize_instructions_magistral
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.utils.hook_utils import add_hooks, get_all_direction_ablation_hooks


class MagistralFactoryTest(unittest.TestCase):
    def test_factory_recognizes_requested_model(self):
        with patch('pipeline.model_utils.magistral_model.MagistralModel') as constructor:
            self.assertIs(construct_model_base('mistralai/Magistral-Small-2506'), constructor.return_value)
            constructor.assert_called_once_with('mistralai/Magistral-Small-2506')


class MagistralIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        try:
            path = hf_hub_download('mistralai/Magistral-Small-2506', 'tekken.json', local_files_only=True)
        except OSError as error:
            raise unittest.SkipTest('Cache the Magistral tekken.json to run native-tokenizer smoke tests') from error
        cls.tokenizer = MistralCommonTokenizer.from_pretrained(str(Path(path).parent), padding_side='left')

    def setUp(self):
        config = MistralConfig(vocab_size=131072, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=8, max_position_embeddings=256, sliding_window=None)
        self.raw = MistralForCausalLM(config).eval()
        self.raw.requires_grad_(False)
        with patch.object(MagistralModel, '_load_model', return_value=self.raw), \
             patch.object(MagistralModel, '_load_tokenizer', return_value=self.tokenizer), \
             patch.object(MagistralModel, '_load_system_prompt', return_value='Be helpful.'):
            self.adapter = construct_model_base('mistralai/Magistral-Small-2506')

    def test_native_prompt_padding_eoi_and_refusal_ids(self):
        inputs = self.adapter.tokenize_instructions_fn(instructions=['red', 'a longer blue instruction'])
        self.assertEqual(inputs.input_ids.shape[0], 2)
        self.assertEqual(inputs.attention_mask[0, 0].item(), 0)
        self.assertEqual(inputs.attention_mask[:, -1].tolist(), [1, 1])
        self.assertEqual(self.adapter.eoi_toks, [4])
        self.assertEqual(inputs.input_ids[:, -1].tolist(), [4, 4])
        self.assertEqual(self.adapter.refusal_toks, [1073, 4072])
        for row, mask in zip(inputs.input_ids, inputs.attention_mask):
            self.assertEqual(row[mask.bool()].tolist().count(self.tokenizer.bos_token_id), 1)
        self.assertEqual(len(self.adapter.model_block_modules), 2)
        self.assertIs(self.adapter.model_attn_modules[0], self.raw.model.layers[0].self_attn)
        self.assertIs(self.adapter.model_mlp_modules[0], self.raw.model.layers[0].mlp)

    def test_output_continuation_preserves_prompt_without_extra_eos(self):
        plain = self.adapter.tokenize_instructions_fn(instructions=['Hello']).input_ids[0].tolist()
        completed = self.adapter.tokenize_instructions_fn(instructions=['Hello'], outputs=['Response']).input_ids[0].tolist()
        self.assertEqual(completed, plain + self.tokenizer.encode('Response', add_special_tokens=False))
        self.assertNotEqual(completed[-1], self.tokenizer.eos_token_id)
        with self.assertRaises(ValueError):
            tokenize_instructions_magistral(self.tokenizer, ['one', 'two'], ['one'])

    def test_generation_respects_explicit_batch_limit(self):
        # Even a batch size cached by an earlier call must respect a smaller limit.
        self.adapter.max_batch_size = 4
        dataset = [{'instruction': text, 'category': 'smoke'} for text in ['red', 'blue']]
        with patch.object(self.raw, 'generate', wraps=self.raw.generate) as generate:
            results = self.adapter.generate_completions(dataset, batch_size=1, max_new_tokens=1)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(call.kwargs['input_ids'].shape[0] == 1 for call in generate.call_args_list))

    def test_actual_mistral_forward_scoring_directions_and_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            directions = generate_directions(self.adapter, ['red'], ['blue'], directory)
        self.assertEqual(tuple(directions.shape), (1, 2, 16))
        self.assertTrue(torch.isfinite(directions).all())
        scores = get_refusal_scores(self.raw, ['red', 'blue'], self.adapter.tokenize_instructions_fn,
                                   self.adapter.refusal_toks, batch_size=2)
        self.assertTrue(torch.isfinite(scores).all())
        pre, post = get_all_direction_ablation_hooks(self.adapter, torch.randn(16))
        inputs = self.adapter.tokenize_instructions_fn(instructions=['Hello'])
        with add_hooks(module_forward_pre_hooks=pre, module_forward_hooks=post):
            logits = self.raw(**inputs).logits
        self.assertTrue(torch.isfinite(logits).all())
        output = self.raw.generate(**inputs, max_new_tokens=1, do_sample=False,
                                   pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        self.assertEqual(output.shape[-1], inputs.input_ids.shape[-1] + 1)


if __name__ == '__main__':
    unittest.main()
