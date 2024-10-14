

class Tokenizer():

    def __init__(self):
        self.start = "^"
        self.end = "$"
        self.pad = 0 # " "
    
    def build_vocab(self, data):
        # Create a set of unique tokens from the data
        unique_tokens = set()
        for sequence in data:
            unique_tokens.update(sequence)
        
        # Convert the set to a sorted list
        self.tokenlist = [self.pad, self.start, self.end] + sorted(unique_tokens)
            
    @property
    def tokenlist(self):
        return self._tokenlist
    
    @tokenlist.setter
    def tokenlist(self, tokenlist):
        self._tokenlist = tokenlist
        # Create the dictionaries      
        self.token_to_int = {c:i for i,c in enumerate(self._tokenlist)}
        self.int_to_token = {i:c for c,i in self.token_to_int.items()}
    
    # def encode(self, sequence):
    #     # Encode the sequence with start and end tokens
    #     return [self.token_to_int[self.start]] + [self.token_to_int[token] for token in sequence] + [self.token_to_int[self.end]]
    
    def encode(self, sequences):
        # Check if the input is a single sequence or list of list sequences
        if len(sequences) == 1:
            # If it's a single sequence, tokenize and encode it
            # sequence = sequences[0]
            return [[self.token_to_int[self.start]] + [self.token_to_int[token] for token in sequences[0]] + [self.token_to_int[self.end]]]
        else:
            # If it's a list of sequences, encode each one
            encoded_sequences = []
            for sequence in sequences:
                # tokens = sequence.split()
                encoded_sequence = [self.token_to_int[self.start]] + [self.token_to_int[token] for token in sequence] + [self.token_to_int[self.end]]
                encoded_sequences.append(encoded_sequence)
            return encoded_sequences


    def decode(self, encoded_sequences):
        # Decode list of sequences
        decoded = []
        for encoded_sequence in encoded_sequences:
            # print(self._tokenlist)
            d = []
            start_pad_end_tokens = [self.token_to_int[self.start], self.token_to_int[self.end], self.token_to_int[self.pad]]

            for ord_val in encoded_sequence:
                if ord_val not in start_pad_end_tokens:
                    d.append(self.int_to_token[ord_val])
            decoded.append(d)


        return decoded

        # Decode the sequence, removing start and end tokens

        # return [self.int_to_token[o] for o in ords if o not in {self.token_to_int[self.start], self.token_to_int[self.end], self.token_to_int[self.pad]}]

    @property
    def n_tokens(self):
        return len(self.int_to_token)



if __name__ == "__main__":
    # tokenizer = Tokenizer()
    # tokenizer.build_vocab([['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']])
    # print(tokenizer.tokenlist)
    # print(tokenizer.encode(['a', 'b', 'c','d']))
    # print(tokenizer.decode(tokenizer.encode(['a', 'b', 'c'])))

    tokenizer = Tokenizer()
    tokenizer.build_vocab([[11, 34], [4, 5, 6], [7, 7]])
    # print(tokenizer.tokenlist)
    # print(tokenizer.encode([[7, 6, 4, 5]]))
    
    encoded = tokenizer.encode([[4, 6, 0], [7, 6, 4]])
    print(encoded)
    print(tokenizer.decode(encoded))

    # *****
    word_tokenizer = Tokenizer()
    word_tokenizer.build_vocab([["hello","how",'are','you']])
    
    print(word_tokenizer.tokenlist)
    print(word_tokenizer.encode([["how",'are','you']]))
    print(word_tokenizer.decode(word_tokenizer.encode([["how",'are','you']])))

    #************ here: How do I take in account that padding adds zero to sequences while padding is encoded as zero in my tokenizer [resolved].