# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer
    trust_remote_code = config.data.get('trust_remote_code', False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    
    #read dataset
    dataset = pd.read_parquet(config.data.path)
    chat_list = dataset[config.data.prompt_key].tolist()
    chat_list = [ chat.tolist() for chat in chat_list]
    #tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #build worker group
    ray_class_with_init = RayClassWithInitArgs(cls = ray.remote(ActorRolloutRefWorker),config=config , role = "rollout")
    resourse_pool = RayResourcePool(process_on_nodes = [config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resourse_pool = resourse_pool , ray_class_with_init = ray_class_with_init)
    wg.init_model()

    #build batches
    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    dispatch_dp_size = wg.world_size
    num_batch = -(-total_samples // config_batch_size )
    
    # 0 is the first layer | 1 to n_sample is the second layer
    n_sample_first_layer  = config.data.n_samples_first_layer
    n_sample_second_layer = config.data.n_samples_second_layer
    assert n_sample_second_layer!=1 , "second samples can't be 1"
    output_layer1 = []
    output_layer2 = []
    for batch_idx in range(num_batch):
        # build batch dict
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_list[ batch_idx * config_batch_size : (batch_idx+1) * config_batch_size ]
        # chat template
        inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                                  add_generation_prompt = True,
                                                  padding = True,
                                                  truncation = True , 
                                                  max_length = config.rollout.prompt_length ,
                                                  return_tensors = "pt",
                                                  return_dict = True,
                                                  tokenize = True)
        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids   = compute_position_id_with_mask(attention_mask)
        batch_dict     = { "input_ids" : input_ids , "attention_mask" : attention_mask , "position_ids" : position_ids }
        #make data & pad
        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch["input_ids"].shape[0]
        if real_batch_size % dispatch_dp_size != 0 :
            dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
            if dummy_data_size <= real_batch_size:
                dummy_data = data[:dummy_data_size]
            else:
                dummy_data = data.repeat( -(-dummy_data_size // real_batch_size) )[:dummy_data_size]
            data = DataProto.concat( [data,dummy_data] )
            print(f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data')    
        batch_size = data.batch["input_ids"].shape[0]
        assert batch_size % dispatch_dp_size == 0 ,f"batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}"
        #generate first step
        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        data.meta_info["num_return_sequences"] = n_sample_first_layer
        # get output
        output = wg.generate_sequences(data)
        #remove_dummy data
        output = output[:real_batch_size * n_sample_first_layer] # sample n times     
        #get response length and split

        #get split
        response_attention_mask = output.batch["attention_mask"][: ,-config.rollout.response_length: ]
        batch_output_length = response_attention_mask.sum(dim=1).tolist()
        batch_output_length = np.array( batch_output_length , dtype = int ).reshape(-1,n_sample_first_layer)
        batch_output_length_min = batch_output_length.min(axis=-1)
        batch_output_length_mean = np.mean(batch_output_length , dim = -1)
        batch_output_split_ratio = np.random.uniform(0.2, 0.8, size=real_batch_size)
        split_index1 = (batch_output_length_mean * batch_output_split_ratio).astype(int)
        split_index2 = (batch_output_length_min * 0.8).astype(int)
        batch_output_split = np.minimum(split_index1, split_index2) 
        batch_output_split = np.repeat(batch_output_split, n_sample_first_layer)
        """        
        response_attention_mask = output.batch["attention_mask"][:, -config.rollout.response_length:]
        batch_output_length = response_attention_mask.sum(dim=1)
        batch_output_length = batch_output_length.view(-1, n_sample_first_layer)
        batch_output_length_min = batch_output_length.min(dim=-1).values
        batch_output_length_mean = batch_output_length.float().mean(dim=-1)
        batch_output_split_ratio = torch.empty_like(batch_output_length_mean).uniform_(0.2, 0.8)
        split_index = (batch_output_length_mean * batch_output_split_ratio).to(dtype=torch.int)
        batch_output_split = torch.clamp(split_index, min=0, max=batch_output_length_min - 1)
        batch_output_split = batch_output_split.repeat_interleave(n_sample_first_layer)
        """
        #get last punct before split
        response_ids = output.batch["input_ids"][:, -config.rollout.response_length:]
        response_ids_first  = response_ids.clone()
        response_ids_second = response_ids.clone()
        for i,split in enumerate(batch_output_split):
            response_ids_first [i,split:] = tokenizer.pad_token_id
            response_ids_second[i,:split] = tokenizer.pad_token_id

        pad_token = tokenizer.pad_token
        # decode text
        decoded_text_first = tokenizer.batch_decode(response_ids_first, skip_special_tokens=False)
        decoded_text_first_unpad = [ text.replace(pad_token,"") for text in decoded_text_first ]
        decoded_text_second = tokenizer.batch_decode(response_ids_second, skip_special_tokens=False)
        decoded_text_second_unpad = [ text.replace(pad_token,"") for text in decoded_text_second ]

        output_layer1.extend(decoded_text_first_unpad)
        output_layer2.extend(decoded_text_second_unpad)

    
    assert len(output_layer1) == total_samples * n_sample_first_layer ,\
        f"len output_layer1 {output_layer1} != total_samples {total_samples} * n_sample_first_layer {n_sample_first_layer} "
    assert len(output_layer2) == total_samples * n_sample_first_layer,\
         f"len output_layer2 {output_layer2} != total_samples {total_samples} * n_sample_first_layer {n_sample_first_layer} "
    
    output_layer1 = np.array(output_layer1 , dtype=object).reshape(total_samples,n_sample_first_layer)
    output_layer2 = np.array(output_layer2  ,dtype=object).reshape(total_samples,n_sample_first_layer,1)
    output_layer2_part = []

    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst      = chat_list[ batch_idx * config_batch_size : (batch_idx+1) * config_batch_size ]
        batch_output_layer1 = output_layer1[ batch_idx * config_batch_size : (batch_idx+1) * config_batch_size ]
        batch_chat_half_response_lst = []
        for chat,responses in zip(batch_chat_lst,batch_output_layer1):
            for response in responses:
                chat_response = chat.copy()
                chat_response.append( { "role":"assistant" , "content":response})
                batch_chat_half_response_lst.append(chat_response)
        
        inputs = tokenizer.apply_chat_template( batch_chat_half_response_lst,
                                                add_generation_prompt = False,# half generation has added prompt
                                                padding = True,
                                                truncation = True , 
                                                max_length = config.rollout.prompt_length + config.rollout.response_length ,#has half response ,longer
                                                return_tensors = "pt",
                                                return_dict = True,
                                                tokenize = True)
        # generate for n_samples second step
        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids   = compute_position_id_with_mask(attention_mask)
        batch_dict     = { "input_ids" : input_ids , "attention_mask" : attention_mask , "position_ids" : position_ids }
        #make data & pad
        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch["input_ids"].shape[0]
        if real_batch_size % dispatch_dp_size != 0 :
            dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
            if dummy_data_size <= real_batch_size:
                dummy_data = data[:dummy_data_size]
            else:
                dummy_data = data.repeat( -(-dummy_data_size // real_batch_size) )[:dummy_data_size]
            data = DataProto.concat( [data,dummy_data] )
            print(f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data')    
        batch_size = data.batch["input_ids"].shape[0]
        assert batch_size % dispatch_dp_size == 0 ,f"batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}"
        #generate first step
        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        data.meta_info["num_return_sequences"] = n_sample_second_layer-1
        # get output
        output = wg.generate_sequences(data)
        #remove_dummy data
        output = output[:real_batch_size * (n_sample_second_layer-1)] # sample n times     
        response_ids_second = output.batch["input_ids"][:, -config.rollout.response_length:]
        decoded_text_second = tokenizer.batch_decode(response_ids_second, skip_special_tokens=False)
        decoded_text_second_unpad = [ text.replace(pad_token,"") for text in decoded_text_second ]
        output_layer2_part.extend(decoded_text_second_unpad)
        #output list 
    assert len(output_layer2_part) == total_samples * n_sample_first_layer * (n_sample_second_layer-1),\
         f"len output_layer2_part {output_layer2_part} wrong "
    
    output_layer2_part = np.array(output_layer2_part,dtype = object).reshape(-1,n_sample_first_layer,n_sample_second_layer-1)
    output_layer2 = np.concatenate([output_layer2,output_layer2_part] , axis = -1)

    dataset["responses_layer1"] = output_layer1
    dataset["responses_layer2"] = output_layer2
    responses = []
    for layer1, layer2 in zip(output_layer1, output_layer2):
        one_sample = []
        for prefix, suffixes in zip(layer1, layer2):
            for cont in suffixes: 
                one_sample.append(prefix + cont)
        responses.append(one_sample)
    dataset["responses"] = response
    for layer1, layer2 in zip(output_layer1, output_layer2):
        for prefix, suffixes in zip(layer1, layer2):
            suffixes.append(prefix)
    return layer2
    # prompts[ n_sample_first[ n_sample_second stage2, one stage1 ] ]

    #return {"layer1" : output_layer1 , "layer2":output_layer2}
if __name__ == '__main__':
    main()
