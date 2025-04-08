import torch
from generate import generate, text_to_token_ids, token_ids_to_text
from configs import GPT_SMALL
from trained_model import get_simply_trained_model

# train  
model, meta = get_simply_trained_model(skip_cache=True)

# testing temps and top-k
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Ever effort moves you", tokenizer=meta["tokenizer"]),
    max_new_tokens=15,
    context_size=GPT_SMALL["context_length"],
    top_k=10,
    temperature=1
)

print("Output text:\n", token_ids_to_text(token_ids, meta["tokenizer"]))

# RUN 1: top_k=10, temp=2
# Ever effort moves you strokes'd Gis that to me up of the nervous- across have of

# RUN 2: top_k=20, temp=2
# Ever effort moves you ridiculous'd Gis that to give he--had a dep I was taken

# RUN 3: top_k=20, temp=0.1
#  Ever effort moves you?"
#
# "I.
#
#"--I was. The when

# RUN 4: top_k=20, temp=1.0
#  Ever effort moves you?"
#
# "I.
#
#"--I was. The when

# RUN 5: top_k=20, temp=0.5
#  Ever effort moves you?"

# "I seen, for--and a dep I said.

# RUN 6: top_k=10, temp=1
# Ever effort moves you strokes'd-chairs surprise to me he was a little of his forward of
