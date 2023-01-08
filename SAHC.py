import math
import os
import logging
from sampling import SteepHC
from editor import RobertaEditor
from model_args import get_args
import torch.multiprocessing as mp
import warnings
from model_utils.functions import set_seed
import datetime
from dateutil import tz
import torch
os.environ['CUDA_VISIBLE_DEVICES']='4'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tzone = tz.gettz()
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mp.set_start_method('spawn', force=True)
    print("spawned")
except RuntimeError:
    pass

def main():
    args = get_args()
    set_seed(args.seed)
    editor = RobertaEditor(args).to(device)
    sahc = SteepHC(args, editor).to(device)
    of_dir = 'results/' + args.output_dir
    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    if args.direction == '0-1': postfix = '0'
    else: postfix = '1'

    filename='data/{}/test.{}'.format(args.dst,postfix)
    with open(filename, 'r', encoding='utf8') as f:
        data = f.readlines()[:]

    bsz = args.bsz
    max_len=args.max_len
    dst=args.dst
    num_batches = math.ceil(len(data) / float(bsz))
    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

    output_file ='{}_{}_seed={}_{}_{}_{}.txt'.\
        format(timestamp,dst,str(args.seed),args.style_mode,str(args.style_weight),args.direction)
    log_txt_path=os.path.join(of_dir, output_file.split('.txt')[0] + '.log')
    print(log_txt_path)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='',
                        filename=log_txt_path,
                        filemode='w',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    word_pairs ={"ca n't": "can not", "wo n't": "will not"}
    logging.info(args)


    def print_es(a,b,c,d):
        print("Early Stopping!")
        logging.info("Early Stopping!")
        print("{} steps, {}\ttotal score:{} {}".format(a + 1, b, c.item(),d.item()))
        logging.info("{} steps, {}\ttotal score:{} {}".format(a + 1, b, c.item(),d.item()))


    with open(of_dir + output_file, 'w', encoding='utf8') as f, mp.Pool(processes=4) as pool:
        for i in range(num_batches):
            batch_data = data[bsz * i:bsz * (i + 1)]

            #preprocessing
            ref_oris = []
            for d in batch_data:
                for k, v in word_pairs.items():
                     d=d.strip().lower().replace(k, v)
                ref_oris.append(d)

            ref_olds=ref_oris.copy()
            state_vec, _ = editor.state_vec(ref_olds)

            break_flag = False
            max_score=0
            step_max_score_list=[0]
            seq_len=[len(line.split()) for line in ref_olds]
            max_seq_len=max(seq_len)

            for step in range(args.max_steps):
                torch.cuda.empty_cache()

                #get the whole candidate list
                ref_news = pool.starmap(editor.edit, [(ref_olds,[ops]*bsz,[positions]*bsz,bsz,max_len) for positions in range(max_seq_len) for ops in [0,1,2]])

                for idx in range(len(ref_news)):
                    # torch.cuda.empty_cache()
                    ref_new_batch_data=ref_news[idx]

                    index, ref_old_score, ref_new_score, new_style_labels,_ \
                        = sahc.acceptance_prob(ref_new_batch_data, ref_olds, ref_oris, state_vec)

                    ref_hat = ref_new_batch_data[index]
                    new_style_label=new_style_labels[index]

                    if ref_new_score>max_score and ref_new_score>ref_old_score:
                        max_score=ref_new_score
                        select_sent = ref_hat

                    # the style is changed!
                    if args.early_stop == True:
                        if (args.direction == '0-1' and new_style_label == 1) or \
                                (args.direction == '1-0' and new_style_label == 0) :
                            select_sent = ref_hat
                            print_es(step,select_sent,ref_old_score,ref_new_score)
                            break_flag = True
                            break

                if max_score>step_max_score_list[step]: #larger than previous max_score
                    print("hill climbing!")
                    logging.info("hill climbing!")
                    ref_olds = [select_sent]
                    step_max_score_list.append(max_score.item())
                else:
                    print("don't climb, stop!")
                    logging.info("don't climb, stop!")
                    break_flag=True

                if break_flag:
                    break

            if break_flag:
                select_sent = select_sent

            logging.info('climb {} steps, the selected sentence is: {}'.format(step+1,select_sent))
            print('climb {} steps, the selected sentence is: {}'.format(step+1,select_sent))

            logging.info('\n')
            f.write(select_sent + '\n')
            f.flush()

if __name__=="__main__":
    main()

