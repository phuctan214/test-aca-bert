# python shard_data.py \
#     --dir /Users/tannp3.aic/Transformer-MGK/dataset \
#     -o /Users/tannp3.aic/Transformer-MGK/dataset/shard \
#     --num_train_shards 256 \
#     --num_test_shards 128 \
#     --frac_test 0.1

#python generate_samples.py \
#    --dir /Users/tannp3.aic/Transformer-MGK/dataset/shard \
#    -o /Users/tannp3.aic/Transformer-MGK/dataset/hdf5/ \
#    --dup_factor 10 \
#    --seed 42 \
#    --vocab_file None \
#    --do_lower_case 1 \
#    --masked_lm_prob 0.15\
#    --model_name vinai/phobert-base \
#    --max_predictions_per_seq 20 \
#    --n_processes 16\
#    --max_seq_length 128 \

# python generate_samples.py \
#     --dir <path_to_shards> \
#     -o /tam/data/bert/bert_input> \
#     --dup_factor 10 \
#     --seed 42 \
#     --vocab_file <path_to_vocabulary_file> \
#     --do_lower_case 1 \
#     --masked_lm_prob 0.15 \ 
#     --max_seq_length 128 \
#     --model_name bert-large-uncased \
#     --max_predictions_per_seq 20 \
#     --n_processes 16