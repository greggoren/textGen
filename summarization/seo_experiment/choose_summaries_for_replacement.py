def read_summaries(summaries_file,input_data_file):
    summary_stats = {}
    indexes = {}
    with open(input_data_file) as input_data:
        inputs = input_data.readlines()
        with open(summaries_file) as summaries_data:
            for i,summary in summaries_data:
                input = inputs[i]
                doc = input.split("\t")[1]
                index = input.split("\t")[2]
