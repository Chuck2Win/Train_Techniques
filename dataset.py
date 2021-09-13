class My_Dataset(object):
    def __init__(self,train_data, val_data, tokenizer, args): #stopword):
        self.train_data = train_data
        self.val_data = val_data
        self.args = args
        self.tokenizer = tokenizer 
        #self.stopword = stopword
        self.labels = {'hate':1,'offensive':2,'none':0}
        
    
    def _remove_stop_words(self, text, stopword): 
        s=''
        for i in stopword:
            s+='%s|'%i
        s = s[:-1]
        return re.sub(s, '', text)

    def _get_sentence_features(self, sentence_a: list, sentence_b:list, pad_seq_length: int):
        tokens_a = [self.tokenizer.cls_token_id] + self.tokenizer.encode(sentence_a,add_special_tokens=False) + [self.tokenizer.sep_token_id]
        tokens_b = self.tokenizer.encode(sentence_b,add_special_tokens=False) + [self.tokenizer.sep_token_id]
        if len(tokens_b)+len(tokens_a)>pad_seq_length:
            tokens_b = self.tokenizer.encode(sentence_b,add_special_tokens=False)[:(pad_seq_length-1-len(tokens_a))] + [self.tokenizer.sep_token_id]
        padded = [self.tokenizer.pad_token_id]*(pad_seq_length-len(tokens_a)-len(tokens_b))
        ids = tokens_a+tokens_b+padded
        token_type_ids = [0]*len(tokens_a) + [1]*(pad_seq_length-len(tokens_a))
        return ids, token_type_ids

    def make_data_set(self, data, sampling):
        Ids = []
        Token_type_ids = []
        Labels = []
        for i in tqdm(data):
            sentence_a, sentence_b, label = i['news_title'],i['comments'],i['hate']
            ids, token_type_ids = self._get_sentence_features(sentence_a, sentence_b, args['seq_len'])
            Ids.append(ids)
            Token_type_ids.append(token_type_ids)
            Labels.append(self.labels[label])
        #return Ids
        Ids  = torch.LongTensor(Ids)
        Token_type_ids = torch.LongTensor(Token_type_ids)
        Labels = torch.LongTensor(Labels)
        sampler = None
        if sampling == 'weighted':
            counts = [(Labels==i).sum() for i in [0,1,2]]
            weights = 1./np.array(counts) 
            samples_weight = torch.FloatTensor([weights[t] for t in Labels])
            sampler = WeightedRandomSampler(samples_weight,len(Ids),replacement=True)
        Attention_masks = Ids.ne(self.tokenizer.pad_token_id).long()
        dataset = TensorDataset(Ids, Attention_masks, Token_type_ids, Labels)
        dataloader = DataLoader(dataset, batch_size = args['batch_size'], sampler = sampler)
        return dataloader 

class BERT_MODEL(nn.Module):
    def __init__(self,args,bert):
        super().__init__()
        self.args = args
        self.bert = bert
        self.fc_layer = nn.Linear(args['hidden_size'],args['n_labels'])

    def forward(self,input_ids, attention_masks, token_type_ids):
        '''
        data shape : batch size, seq_len
        output shape : batch size, n_labels
        '''

        out = self.bert(input_ids, attention_masks, token_type_ids)
        # CLS에 tanh() 씌운 것임
        o = out.pooler_output
        
        #o_pool = nn.Tanh()(o)
        output = self.fc_layer(o)
        return output      
