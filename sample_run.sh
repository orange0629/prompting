## declare an array variable
declare -a models=("/shared/4/models/llama2/pytorch-versions/llama-2-7b-chat")
declare -a benchmarks=("socket_bragging#brag_achievement" "socket_hahackathon#is_humor" "socket_tweet_irony" "socket_sexyn" "socket_tweet_offensive" "socket_complaints" "socket_empathy#empathy_bin" "socket_stanfordpoliteness" "socket_rumor#rumor_bool")

## now loop through the above array
for i in "${models[@]}"
do
    for j in "${benchmarks[@]}"
    do
        echo "Now running $j on $i"
        #CUDA_VISIBLE_DEVICES=1 python scripts/inference_vllm.py -model_dir $i -benchmark $j -system_prompts_dir "./data/system_prompts/Prompt-Scores_Baseline.csv" -saving_strategy "eval" -cache_dir "/shared/4/models/"
        #CUDA_VISIBLE_DEVICES=1 python scripts/inference_vllm.py -model_dir $i -benchmark $j -system_prompts_dir "./data/system_prompts/Prompt-Scores_Good-Property.csv" -cache_dir "/shared/4/models/"
        #CUDA_VISIBLE_DEVICES=1 python scripts/inference_vllm.py -model_dir $i -benchmark $j -system_prompts_dir "./data/system_prompts/Prompt-Scores_Good-Property.csv" -cache_dir "/shared/4/models/"
    done
done