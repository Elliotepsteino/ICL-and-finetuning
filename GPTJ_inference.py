from transformers import AutoTokenizer,GPTJForCausalLM,AutoModelForCausalLM
import torch
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

modelname ="EleutherAI/gpt-j-6B"
modelname ="togethercomputer/GPT-JT-6B-v1"
# load fp 16 model
#model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16)
#model = AutoModelForCausalLM.from_pretrained(modelname)
# save model with torch.save
#breakpoint()
#breakpoint()
#torch.save(model, "gptjt.pt")
if modelname =="togethercomputer/GPT-JT-6B-v1":

    model = torch.load("gptjt.pt")
else:
    model =torch.load("gptj.pt")
tokenizer = AutoTokenizer.from_pretrained(modelname)


#model = GPTJForCausalLM.from_pretrained(
#    "EleutherAI/gpt-j-6B",
##        revision="float16",
#        torch_dtype=torch.float16,
#        low_cpu_mem_usage=True
#)

model.cuda()
model.eval()

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
examples_to_print = 50
context_examples=3
device ='cuda'
genlen = 50
text = b" Answer True or False: "
#text= b" "
end_of_example = b" <|endoftext|> "
end_of_example = b" "
for i in range(0,examples_to_print*context_examples,context_examples):
    try:
        prompt = b""
        for j in range(context_examples):
            if j==context_examples-1:
                prompt += (questions[i+j]) + text
            else:
                prompt += (questions[i+j]) + text + (answers[i+j]) + end_of_example
        
        prompt = str(prompt)
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device=device)

        max_length = input_ids.shape[1] + genlen

        output_ids = model.generate(input_ids=input_ids, max_length=max_length, 
                            do_sample=True,eos_token_id=tokenizer.eos_token_id)
        
        answer_to_question=tokenizer.decode(output_ids[0][input_ids.shape[1]:])
        print("Q: ", str(prompt))
        print("Ans: ", str(answer_to_question))
        print("Correct Ans: ", str(answers[i+context_examples-1]))
    except:
        print("\nCould not parse\n")
        continue