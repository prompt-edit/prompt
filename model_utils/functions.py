import random
import numpy as np
import torch
from nltk.corpus import stopwords
from model_args import get_args
args=get_args()
stopwords.words('english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

positive_word_lst=['positive','good','better','well']
negative_word_lst=['negative','bad','worse','depressed']
formal_word_lst=['formal']
informal_word_lst=['informal']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x):
    x = x - torch.max(x)
    exp_x = torch.exp(x)
    softmax_x = exp_x / torch.sum(exp_x)
    return softmax_x

#gpt-j-6b
    # it 's small and they make you feel not right at home
    # positive score: 0.71
    # negative score: 0.29 --> 0.68 ; 0.01--> 0.99

    # 0.6-> 6 -> 36
    # 0.4-> 4 -> 16
def predict_next_word(model,tokenizer,input_text,direction):

    tokens_tensor = {k: v.to(device) for k, v in tokenizer(input_text, padding=True, return_tensors="pt").items()}
    # Set the model in evaluation mode to deactivate the DropOut modules
    model.eval()
    # If you have a GPU, put everything on cuda
    # Predict all tokens
    with torch.no_grad():
      outputs = model(**tokens_tensor)
      predictions = outputs[0]

    # Get the predicted next sub-word
    # if [0, -1, :] --> dim_size (1, 50257); if [:, -1, :] --> (50257,)
    probs = torch.tensor(predictions[:, -1, :],dtype=torch.float32)

    dst = args.dst
    if dst == 'yelp' or dst == 'amazon':
        pos_logits = probs[:,tokenizer.encode('positive')]
        neg_logits = probs[:,tokenizer.encode('negative')]
    elif dst=='gyafc':
        pos_logits = probs[:,tokenizer.encode(' formal')] # 1--> informal
        neg_logits = probs[:,tokenizer.encode(' informal')]
    elif dst=='jfleg':
        pos_logits = probs[:, tokenizer.encode('logical')]
        neg_logits = probs[:, tokenizer.encode('errors')]

    elif dst=='shakespeare':
        pos_logits = probs[:, tokenizer.encode('modern')]
        neg_logits = probs[:, tokenizer.encode('Elizabeth')]
    elif dst =='sym':
        pos_logits = probs[:, tokenizer.encode(' symbolic')]
        neg_logits = probs[:, tokenizer.encode(' English')]

    emo_logits = torch.dstack((neg_logits,pos_logits)).squeeze(1)
    softmax_emo_logits = torch.softmax(emo_logits,dim=1)

    neg_prob = softmax_emo_logits[:, 0]
    pos_prob = softmax_emo_logits[:, 1]

    if direction=='0-1':
        output_prob = pos_prob / neg_prob  # make the prob more robust
    else: #1-0
        output_prob = neg_prob / pos_prob

    # if direction=='0-1':
    #     output_prob = pos_prob
    # else: #1-0
    #     output_prob = neg_prob

    dst = args.dst

    if args.setting=='zero-shot':
        if dst == 'yelp':
            thres_neg=0.6
            thres_pos=0.9
        elif dst == 'amazon':
            thres_neg=0.7
            thres_pos=0.9
        elif dst == 'gyafc':
            thres_neg=0.9
            thres_pos=0.85
        elif dst =='jfleg' or dst =='shakespeare':
            thres_pos=0.9
            thres_neg = 0.9


    else: # few-shot
        thres_neg=0.9
        thres_pos=0.7


    emo_argmax_labels=torch.argmax(softmax_emo_logits,dim=1)
    labels = []
    for idx in range(len(softmax_emo_logits)):
        if emo_argmax_labels[idx] == 0 and neg_prob[idx]>=thres_neg:
            labels.append(0) # 'negative'
        elif emo_argmax_labels[idx] == 1 and pos_prob[idx]>=thres_pos:
            labels.append(1) # 'positive'
        else: labels.append(2) # 'neutral'


    return output_prob, labels

#huggingface classifier
def pipe(res_cand,direction):
    label = res_cand[0]['label'].lower()
    score = res_cand[0]['score']

    if direction=='0-1':
        classifi_prob = score if label == 1 else 1- score
    else: # 1-0
        classifi_prob= score if label == 0 else 1-score

    return classifi_prob,label

def pytorch_cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
