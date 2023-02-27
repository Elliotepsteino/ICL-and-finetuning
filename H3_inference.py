import tensorflow as tf
import argparse

import torch
import os
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

from transformers import GPT2Tokenizer

from H3.src.models.ssm_seq import SSMLMHeadModel

##load_data loads data from a tfrecord file
#Called /data/P3/data/super_glue_boolq_exercise/train.tfrecord-00000-of-00001 and 
#for each example in the dataset, append it to a numpy array
#Then, return the numpy array
def load_data(filename):
    dataset = tf.data.TFRecordDataset(filename)
    def _parse_function(example_proto):
        features = { 
                    "inputs_pretokenized": tf.io.FixedLenFeature([], tf.string),
                    "targets_pretokenized": tf.io.FixedLenFeature([], tf.string)}
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return parsed_features["targets_pretokenized"], parsed_features["inputs_pretokenized"]
    dataset = dataset.map(_parse_function)
    answers = []
    questions = []
    for i in dataset:
        answers.append(i[0].numpy())
        questions.append(i[1].numpy())


    return answers, questions

answers, questions = load_data("/data/P3/data/super_glue_boolq_exercise/train.tfrecord-00000-of-00001")
parser = argparse.ArgumentParser(description='H3 text generation')
parser.add_argument('--dmodel', type=int, default=2048)
parser.add_argument('--nlayer', type=int, default=24)
parser.add_argument('--attn-layer-idx', nargs='+', type=int, default=[8,16])
parser.add_argument('--rotary_emb_dim', type=int, default=None, help='For rotary embeddings, set to 64. Default is None.')
parser.add_argument('--nheads', type=int, default=16)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--genlen', type=int, default=128)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--prompt', type=str, default='Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that')
args = parser.parse_args()

device = 'cuda'
dtype = torch.float16
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
torch.random.manual_seed(0)
d_model = args.dmodel
n_layer = args.nlayer
ssm_cfg = dict(mode='diag', measure='diag-lin')
attn_layer_idx = args.attn_layer_idx

if args.rotary_emb_dim is None:
    attn_cfg = dict(num_heads=args.nheads)
else:
    attn_cfg = dict(num_heads=args.nheads, rotary_emb_dim=args.rotary_emb_dim)
print("Loading model...")
model = SSMLMHeadModel(d_model, n_layer=n_layer, d_inner=4 * d_model, vocab_size=len(tokenizer),
                       ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                       pad_vocab_size_multiple=8).to(device=device)
#args.ckpt=None
print("Loading checkpoint...")
if args.ckpt is not None:
    state_dict = torch.load(args.ckpt, map_location="cpu")
    if 'pytorch-lightning_version' in state_dict:
        state_dict = {k[len('model.'):]: v for k, v in state_dict['state_dict'].items()
                      if k.startswith('model.')}
    model.load_state_dict(state_dict)

model.cuda()
model.eval()
# Only cast the nn.Linear parameters to dtype, the SSM params stay in fp32
# Pytorch lacks support for complex32 (i.e. complex<float16>) and complex<bfloat16>.

for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
        module.to(dtype=dtype)

#tokbreakpoint()
examples_to_print = 50
context_examples=1
for i in range(0,examples_to_print*context_examples,context_examples):
    try:
        prompt = b""
        for j in range(context_examples):
            if j==context_examples-1:
                prompt += (questions[i+j]) + b" Answer True or False: "
            else:
                prompt += (questions[i+j]) + b" Answer True or False: " + (answers[i+j]) + b" <|endoftext|> "
        prompt = str(prompt)
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device=device)

        max_length = input_ids.shape[1] + args.genlen

        output_ids = model.generate(input_ids=input_ids, max_length=max_length,
                            return_dict_in_generate=False, output_scores=False, 
                            timing=False, top_p=args.top_p, top_k=args.top_k, 
                            eos_token_id=tokenizer.eos_token_id)
        
        answer_to_question=tokenizer.decode(output_ids[0][input_ids.shape[1]:])
        print("Q: ", str(prompt))
        print("Ans: ", str(answer_to_question))
        print("Correct Ans: ", str(answers[i+context_examples-1]))
    except:
        print("\nCould not parse\n")
        continue

    






    
