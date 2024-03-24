#a gradio interface for running the model and seeing the token speed
import itertools, sys, time, argparse, inspect
from pathlib import Path 
from typing import Optional, Tuple 
import gradio as gr
from fastchat.model import get_conversation_template

import torch
import torch._dynamo.config 
import torch._inductor.config 

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

#compilation arguments 
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
torch._inductor.config.triton.cudagraph_trees = False #maybe there is a bug in the cudagraph trees

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sentencepiece import SentencePieceProcessor

from model import Transformer

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits.to(dtype = torch.float32) / max(temperature, 1e-5)

    # print("Datatype of logits: ", logits.dtype)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    # print("Logits are ", logits)
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    # print("Probs are ", probs)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]d
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)
            yield new_tokens

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested. No speculative decoding
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
    final_tokens = []
    for token_sets in generated_tokens:
        final_tokens = token_sets
        yield torch.cat(token_sets) #yield the tokens
        
    seq[T + 1:] = torch.cat(final_tokens)

    return seq

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-3].startswith("g")
        assert path_comps[-2] in device, "weight packed format mismatch, please rerun quantize.py!"
        groupsize = int(path_comps[-3][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime(use_cuda)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    # model = model.to(device=device, dtype = precision)
    model = model.to(device = device)
    return model.eval()

#gradio helper functions from here <---------------------->
def user(user_message, history,session_state):
    if history==None:
        history=[]
    pure_history = session_state.get("pure_history", [])
    pure_history += [[user_message, None]]
    session_state["pure_history"] = pure_history
    return "", history + [[user_message, None]],session_state

def clear(history,session_state):
    pure_history = session_state.get("pure_history", [])
    pure_history = []
    session_state["pure_history"] = pure_history
    return [],"0.00 tokens/s","0.00",session_state

def regenerate(history,session_state):
    if not history:
        return history, None,"0.00 tokens/s","0.00",session_state
    pure_history = session_state.get("pure_history", [])
    pure_history[-1][-1] = None
    session_state["pure_history"]=pure_history
    if len(history) > 1:  # Check if there's more than one entry in history (i.e., at least one bot response)
        new_history = history[:-1]  # Remove the last bot response
        last_user_message = history[-1][0]  # Get the last user message
        return new_history + [[last_user_message, None]], None,"0.00 tokens/s","0.00",session_state
    history[-1][1] = None
    return history, None,"0.00 tokens/s","0.00",session_state

def bot(
    history, 
    session_state,
): 
    temperature, top_p, use_EaInfer, highlight_EaInfer =0.7,0.0,True,True
    #we will ignore highlight_EaInfer for now 
    global args
    if not history:
        return history, "0.00 tokens/s", "0.00", session_state
    pure_history = session_state.get("pure_history", [])
    assert args.model_type == "llama-2-chat" or "vicuna"
    conv = get_conversation_template(args.model_type)

    if args.model_type == "llama-2-chat":
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
    elif args.model_type == "mixtral":
        conv = get_conversation_template("llama-2-chat")
        conv.system_message = ''
        conv.sep2 = "</s>"

    for query, response in pure_history:
        conv.append_message(conv.roles[0], query)
        if args.model_type == "llama-2-chat" and response:
            response = " " + response
        conv.append_message(conv.roles[1], response)

    prompt = conv.get_prompt()

    if args.model_type == "llama-2-chat":
        prompt += " "

    print("Prompt is ", prompt)

    global tokenizer, model 

    encoded = encode_tokens(tokenizer, prompt, bos = True, device = default_device)
    prompt_length = encoded.size(0)

    device_sync(device = default_device)
    t0 = time.perf_counter()
    done_generating = False
    buffer = ''
    period_id = tokenizer.encode('.')[0]

    y = generate(
        model, 
        encoded,
        max_new_tokens = 1024,
        temperature = temperature,
    )

    for tokens in y:
        t1 = time.perf_counter()
        history[-1][1] = tokenizer.decode([period_id] + tokens.tolist())[1:]
        yield history, f"{len(tokens.tolist()) / (t1 - t0):.2f} tokens/s", session_state


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
parser.add_argument('--eagle_checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/eagle.pth"), help='Eagle checkpoint path.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument("--model-type", type=str, default="llama-2-chat", help="llama-2-chat or vicuna, for chat template")
args = parser.parse_args()

assert args.checkpoint_path.is_file(), args.checkpoint_path
tokenizer_path = args.checkpoint_path.parent / "tokenizer.model"
assert tokenizer_path.is_file(), tokenizer_path

from tp import maybe_init_dist 
rank = maybe_init_dist()

use_tp = rank is not None 
if use_tp: 
    if rank != 0:
        #replace print for ranks other than 0
        print = lambda *args, **kwargs: None

precision = torch.bfloat16
print("Loading model...", flush = True)

start_time = time.time()
model = _load_model(args.checkpoint_path, default_device, precision, use_tp)
device_sync(device = default_device)
endd_time = time.time()
print(f"Loading model took {endd_time - start_time}s")

tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

#compile the forward pass 
if args.compile: 
    print("Compile decode one token")
    decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
    print("Compiling decode one token done")

    print("Compiling prefill...", flush = True)
    prefill = torch.compile(prefill, mode="reduce-overhead", fullgraph=True)
    print("Prefill compiled!", flush = True)

custom_css = """
#speed textarea {
    color: red;   
    font-size: 30px; 
}"""

#launch the gradio only on rank 0? not sure how to do that 
with gr.Blocks(css=custom_css) as demo:
    gs = gr.State({"pure_history": []})
    gr.Markdown('''## EAGLE - gptfast Chatbot''')
    with gr.Row():
        speed_box = gr.Textbox(label="Speed", elem_id="speed", interactive=False, value="0.00 tokens/s")


    chatbot = gr.Chatbot(height=600,show_label=False)

    msg = gr.Textbox(label="Your input")
    with gr.Row():
        send_button = gr.Button("Send")
        stop_button = gr.Button("Stop")
        regenerate_button = gr.Button("Regenerate")
        clear_button = gr.Button("Clear")
    enter_event=msg.submit(user, [msg, chatbot,gs], [msg, chatbot,gs], queue=True).then(
        bot, [chatbot,gs], [chatbot,speed_box,gs]
    )
    clear_button.click(clear, [chatbot,gs], [chatbot,speed_box,gs], queue=True)

    send_event=send_button.click(user, [msg, chatbot,gs], [msg, chatbot,gs],queue=True).then(
        bot, [chatbot ,gs], [chatbot,speed_box,gs]
    )
    regenerate_event=regenerate_button.click(regenerate, [chatbot,gs], [chatbot, msg,speed_box,gs],queue=True).then(
        bot, [chatbot,gs], [chatbot,speed_box,gs]
    )
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[send_event,regenerate_event,enter_event])
demo.queue()
demo.launch(share=True)