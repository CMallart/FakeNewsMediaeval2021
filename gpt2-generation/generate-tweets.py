from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import torch
import argparse


class TextGenerator():

    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

        # I'm not really doing anything with the config buheret
        self.configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

        # instantiate the model
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=self.configuration)

        # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
        # otherwise the tokenizer and model tensors won't match up
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.load_state_dict(torch.load(model_path))


    def generate(self, prompt, num_tweets, output_file, output_type="txt", temperature=1.0):
        self.model.eval()
        device = torch.device("cpu")

        prompt = "<|startoftext|>"+str(prompt)

        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(device)

        print("Starting generation of {} tweets".format(num_tweets))
        sample_outputs = self.model.generate(
                                        generated, 
                                        #bos_token_id=random.randint(1,30000),
                                        do_sample=True,   
                                        top_k=50, 
                                        max_length = 300,
                                        top_p=0.95, 
                                        temperature = temperature,
                                        repetition_penalty =0.5, 
                                        num_return_sequences=num_tweets
                                        )
        print("Tweets generated")
        if output_type=="txt":
            with open(output_file, "a+") as outfile:
                for i, sample_output in enumerate(sample_outputs):
                    tweet= "{}: {}\n".format(i, self.tokenizer.decode(sample_output, skip_special_tokens=True))
                    outfile.write(tweet)
        print("Tweets saved to {}".format(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate tweets.')
    parser.add_argument('--model_path', type=str, help='path to the trained GPT-2 model')
    parser.add_argument('--output_type', default="txt", type=str,
                        help='how to output the generated tweets : txt=a txt with one tweet per line')
    parser.add_argument('--output_file', default="txt", type=str,
                        help='where to output the generated tweets')
    parser.add_argument('--num_tweets', default="10", type=int,
                        help='number of tweets to generate')
    parser.add_argument('--prompt', default="", type=str,
                        help='prompt to give to generate the tweets')
    parser.add_argument('--temperature', default="1.0", type=float,
                        help='temperature for generation')
    args = parser.parse_args()

    
    generator = TextGenerator(args.model_path)
    generator.generate(args.prompt, args.num_tweets, args.output_file, args.output_type, args.temperature)
