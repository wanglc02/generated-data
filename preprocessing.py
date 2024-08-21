def preprocess_qqp(tokenizer, dataset, max_length=128):
    return _generic_preprocess(tokenizer, dataset, max_length, ['question1', 'question2'])

def preprocess_cola(tokenizer, dataset, max_length=128):
    return _generic_preprocess(tokenizer, dataset, max_length, ['sentence'])

def preprocess_imdb(tokenizer, dataset, max_length=512):
    return _generic_preprocess(tokenizer, dataset, max_length, ['text'])

def preprocess_mrpc(tokenizer, dataset, max_length=128):
    return _generic_preprocess(tokenizer, dataset, max_length, ['sentence1', 'sentence2'])

def preprocess_qnli(tokenizer, dataset, max_length=128):
    return _generic_preprocess(tokenizer, dataset, max_length, ['question', 'sentence'])

def preprocess_rte(tokenizer, dataset, max_length=128):
    return _generic_preprocess(tokenizer, dataset, max_length, ['sentence1', 'sentence2'])

def preprocess_sst2(tokenizer, dataset, max_length=128):
    return _generic_preprocess(tokenizer, dataset, max_length, ['sentence'])

def preprocess_yelp(tokenizer, dataset, max_length=512):
    return _generic_preprocess(tokenizer, dataset, max_length, ['text'])

def _generic_preprocess(tokenizer, dataset, max_length, text_fields):
    def tokenize_function(example):
        texts = (example[field] for field in text_fields)
        return tokenizer(*texts, padding="max_length", truncation=True, max_length=max_length)
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset
