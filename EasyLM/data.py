import time
from functools import partial
import json
import base64
from multiprocessing import Pool

from tqdm import tqdm
import mlxu
from ml_collections import ConfigDict
import numpy as np
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import numpy_default_data_collator

class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()
        config.json_torch_dataset = JsonTorchDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        text_processor = TextProcessor(config.text_processor, tokenizer)
        if config.type == 'huggingface':
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        elif config.type == 'json_torch':
            torch.manual_seed(42)
            dataset = JsonTorchDataset(config.json_torch_dataset, tokenizer, text_processor, **kwargs)
            return DataLoader(
                dataset,
                batch_size=config.json_torch_dataset.batch_size,
                num_workers=config.json_torch_dataset.num_workers,
                shuffle=True,
                collate_fn=numpy_default_data_collator,
                drop_last=True  # sometimes batch doesnt split across tpu well.
            )
        elif config.type == 'tulu_json_torch':
            torch.manual_seed(42) # keep dataloader order the same across devices.
            dataset = TuluJsonTorchDataset(config.json_torch_dataset, tokenizer, text_processor, **kwargs)
            return DataLoader(
                dataset,
                batch_size=config.json_torch_dataset.batch_size,
                num_workers=config.json_torch_dataset.num_workers,
                shuffle=True,
                collate_fn=numpy_default_data_collator,
                drop_last=True  # sometimes batch doesnt split across tpu well.
            )
        elif config.type == 'preference_json_torch':
            torch.manual_seed(42)
            dataset = PreferenceDataset(config.json_torch_dataset, tokenizer, text_processor, **kwargs)
            return DataLoader(
                dataset,
                batch_size=config.json_torch_dataset.batch_size,
                num_workers=config.json_torch_dataset.num_workers,
                shuffle=True,
                collate_fn=numpy_default_data_collator,
                drop_last=True  # sometimes batch doesnt split across tpu well.
            )
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        config.base64_token_dtype = 'i4'
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []

        if self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        prev_text = ''
        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field.startswith('<|') and field.endswith('|>'):
                # Special tokens.
                field = field[2:-2]
                if field == 'bos':
                    token_buffer.append(self.tokenizer.bos_token_id)
                elif field == 'eos':
                    token_buffer.append(self.tokenizer.eos_token_id)
                else:
                    # Token ID specified directly.
                    token_buffer.append(int(field))
                loss_mask_buffer.append(mask)
            elif field.startswith('{') and field.endswith('}'):
                field = field[1:-1]
                # Base64 encoded raw tokens.
                tokens = np.frombuffer(
                    base64.b64decode(example[field]),
                    dtype=self.config.base64_token_dtype
                ).tolist()
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                if i > 0 and not prev_text.endswith((' ', '\n', '\t')):
                    text = ' ' + text.strip()
                tokens = self.tokenizer.encode(text)
                prev_text = text
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, *aux


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.batch_token_dtype = 'i4'

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    input_tokens = np.array(token_buffer[:chunk_size], dtype=self.config.batch_token_dtype).reshape(
                        self.config.batch_size, -1
                    )
                    attention_mask = np.ones_like(input_tokens)
                    # mask out the padding
                    attention_mask[input_tokens == self.tokenizer.pad_token_id] = 0
                    batch = {
                        'input_tokens': input_tokens,
                        'attention_mask': attention_mask,
                        'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    if self.config.always_start_with_bos:
                        batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                        batch['attention_mask'][:, 0] = 1
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))

    # write a function to get the length of the dataset
    def __len__(self):
        return len(self._dataset)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:   # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index in self.parallel_example_iterator():
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (
                    (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)

    def _finite_json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            for line in fin:
                if not line or line == '\n':
                    continue
                try:
                    data = json.loads(line)
                except json.decoder.JSONDecodeError:
                    print(f'Error parsing json line:\n{line}')
                    continue
                yield data


    def __len__(self):
        return sum(1 for _ in self._finite_json_iterator())

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)


class JsonTorchDataset(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.num_workers = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        # self.dataset = [x for x in tqdm(self._load_file(), desc='Loading Dataset')]
        if 'gs://' in self.config.path:
            dataset = []
            with mlxu.open_file(self.config.path, 'r') as fin:
                for line in tqdm(fin, desc='Loading Dataset'):
                    if not line or line == '\n':
                        continue
                    try:
                        data = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        print(f'Error parsing json line:\n{line}')
                        continue
                    dataset.append(data)
            # load into huggingface dataset 
            dataset = Dataset.from_list(dataset)
            # dataset = dataset.shard(num_shards=1000, index=0)
            self.dataset = dataset.map(
                self._process_sample,
                batched=False,
                num_proc=self.config.num_workers,
                remove_columns=[x for x in dataset.column_names if x not in ['input_tokens', 'target_tokens', 'loss_masks', 'attention_mask']],)
            
        else:
            dataset = load_dataset('json', data_files=self.config.path)
            # dataset['train'] = dataset['train'].shard(num_shards=1000, index=0)
            self.dataset = dataset['train'].map(
                self._process_sample,
                batched=False,
                num_proc=self.config.num_workers,
                remove_columns=[x for x in dataset['train'].column_names if x not in ['input_tokens', 'target_tokens', 'loss_masks', 'attention_mask']],)
        
    def _json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            for line in fin:
                if not line or line == '\n':
                    continue
                try:
                    data = json.loads(line)
                except json.decoder.JSONDecodeError:
                    print(f'Error parsing json line:\n{line}')
                    continue
                yield data

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def _process_sample(self, sample):
        tokens = self.tokenizer.encode(sample['prompt'] + sample['completion'])
        tokens = tokens[:self.config.seq_length]
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        prompt_len = len(self.tokenizer.encode(sample['prompt'])) + 1  # add bos token
        loss_masks = ([0.0] * prompt_len) + ([1.0] * (len(tokens) - prompt_len))
        # trunacte and pad everything out
        if len(tokens) > self.config.seq_length:
            tokens = tokens[:self.config.seq_length]
            loss_masks = loss_masks[:self.config.seq_length]
        # before padding, account for shifting
        input_tokens = tokens[:-1]
        loss_masks = loss_masks[1:]
        target_tokens = tokens[1:]
        attention_mask = [1] * len(input_tokens) + [0] * (self.config.seq_length - len(input_tokens))
        input_tokens = input_tokens + [self.tokenizer.pad_token_id] * (self.config.seq_length - len(input_tokens))
        target_tokens = target_tokens + [self.tokenizer.pad_token_id] * (self.config.seq_length - len(target_tokens))
        loss_masks = loss_masks + [0.0] * (self.config.seq_length - len(loss_masks))
        return {
            "input_tokens": np.array(input_tokens, dtype=np.int32),
            "target_tokens": np.array(target_tokens, dtype=np.int32),
            "loss_masks": np.array(loss_masks, dtype=np.float32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
        }

    def __len__(self):
        return len(self.dataset)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)


class TuluJsonTorchDataset(JsonTorchDataset):

    def _process_sample(self, sample):
        # run tulu processor
        tokens, labels, attention_mask = self.encode_with_messages_format(sample, self.tokenizer, self.config.seq_length)
        loss_masks = [1.0 if x != -100 else 0.0 for x in labels]
        # before padding, account for shifting
        input_tokens = tokens[:-1].tolist()
        attention_mask = attention_mask[:-1].tolist()
        loss_masks = loss_masks[1:]
        target_tokens = tokens[1:].tolist()
        # pad everything out
        attention_mask = attention_mask + [0] * (self.config.seq_length - len(attention_mask))
        input_tokens = input_tokens + [self.tokenizer.pad_token_id] * (self.config.seq_length - len(input_tokens))
        target_tokens = target_tokens + [self.tokenizer.pad_token_id] * (self.config.seq_length - len(target_tokens))
        loss_masks = loss_masks + [0.0] * (self.config.seq_length - len(loss_masks))
        return {
            "input_tokens": np.array(input_tokens, dtype=np.int32),
            "target_tokens": np.array(target_tokens, dtype=np.int32),
            "loss_masks": np.array(loss_masks, dtype=np.float32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
        }

    def encode_with_messages_format(self, example, tokenizer, max_seq_length):
        messages = example['messages']
        if len(messages) == 0:
            raise ValueError('messages field is empty.')
        
        def _concat_messages(messages):
            message_text = ""
            for message in messages:
                if message["role"] == "system":
                    message_text += "<|system|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "user":
                    message_text += "<|user|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "assistant":
                    message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
            return message_text
            
        example_text = _concat_messages(messages).strip()
        example_text = tokenizer.bos_token + example_text
        tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(messages[:message_idx+1])
                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors='pt', 
                    max_length=max_seq_length, 
                    truncation=True
                ).input_ids.shape[1]
                # we have to add bos offset
                labels[:, message_start_idx+1:message_end_idx+1] = -100
                
                if message_end_idx >= max_seq_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        return input_ids.flatten(), labels.flatten(), attention_mask.flatten()
    

# for processing preference-style datasets
# expect: a jsonl file with each line being a json object with the following fields:
#   - prompt: the initial prompt **with whitespace at the end**
#   - chosen: the chosen completion
#   - rejected: the rejected completion
class PreferenceDataset(JsonTorchDataset):

    def _process_sample(self, sample):
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        # tokenize the prompt with chosen and rejected, truncate to seq_length
        chosen_input_ids = self.tokenizer(prompt + chosen, max_length=self.config.seq_length, truncation=True).input_ids
        rejected_input_ids = self.tokenizer(prompt + rejected, max_length=self.config.seq_length, truncation=True).input_ids
        chosen_attn_mask = [1] * len(chosen_input_ids)
        rejected_attn_mask = [1] * len(rejected_input_ids)
        # setup loss mask for chosen and rejected
        num_prompt_tokens = len(self.tokenizer(prompt, max_length=self.config.seq_length, truncation=True).input_ids)
        chosen_loss_mask = [0.0] * num_prompt_tokens + [1.0] * (len(chosen_input_ids) - num_prompt_tokens)
        rejected_loss_mask = [0.0] * num_prompt_tokens + [1.0] * (len(rejected_input_ids) - num_prompt_tokens)
        # pad everything out
        chosen_attn_mask = chosen_attn_mask + [0] * (self.config.seq_length - len(chosen_attn_mask))
        rejected_attn_mask = rejected_attn_mask + [0] * (self.config.seq_length - len(rejected_attn_mask))
        chosen_input_ids = chosen_input_ids + [self.tokenizer.pad_token_id] * (self.config.seq_length - len(chosen_input_ids))
        rejected_input_ids = rejected_input_ids + [self.tokenizer.pad_token_id] * (self.config.seq_length - len(rejected_input_ids))
        chosen_loss_mask = chosen_loss_mask + [0.0] * (self.config.seq_length - len(chosen_loss_mask))
        rejected_loss_mask = rejected_loss_mask + [0.0] * (self.config.seq_length - len(rejected_loss_mask))
        return {
            "chosen_input_ids": np.array(chosen_input_ids, dtype=np.int32),
            "chosen_loss_mask": np.array(chosen_loss_mask, dtype=np.float32),
            "chosen_attn_mask": np.array(chosen_attn_mask, dtype=np.int32),
            "rejected_input_ids": np.array(rejected_input_ids, dtype=np.int32),
            "rejected_loss_mask": np.array(rejected_loss_mask, dtype=np.float32),
            "rejected_attn_mask": np.array(rejected_attn_mask, dtype=np.int32),
        }


if __name__ == "__main__":
    from EasyLM.models.llama.llama_model import LLaMATokenizer
    tokenizer = LLaMATokenizer(
        vocab_file='gs://hamishi-dev/easylm/llama/tokenizer.model',
        add_bos_token=False,
        add_eos_token=False,
        padding_side='left',
        truncation_side='right',
    )
    text_processor = TextProcessor({'fields': '[prompt],completion'}, tokenizer)
    dataset = TuluJsonTorchDataset(TuluJsonTorchDataset.get_default_config({'path': 'tulu_v2_mix.jsonl'}), tokenizer, text_processor)
    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
        shuffle=True,
        collate_fn=numpy_default_data_collator,
        drop_last=True  # sometimes batch doesnt split across tpu well.
    )
    for sample in loader:
        import pdb; pdb.set_trace()