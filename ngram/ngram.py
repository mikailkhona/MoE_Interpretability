from collections import defaultdict
from itertools import product
import random
import pickle
from tqdm import tqdm
import numpy as np

class NGram:
    def __init__(self, vocab, n, prob_table=None):
        self.vocab = vocab
        self.n = n
        if prob_table: # smaller ngram
            self.prob_table = prob_table
            self.smaller_ngrams = None
        else: # main ngram
            self.prob_table = self._initialize_prob_table()
            self.smaller_ngrams = self._recursive_smaller_ngram()

    def _initialize_prob_table(self):
        prob_table = defaultdict(dict)
        for context in self._generate_contexts():
            total_prob = 0
            for token in self.vocab:
                prob = random.random()
                prob_table[context][token] = prob
                total_prob += prob
            
            # Normalize probabilities
            for token in self.vocab:
                prob_table[context][token] /= total_prob
        
        return prob_table

    def _generate_contexts(self):
        if self.n == 1:
            return [()]
        return list(product(self.vocab, repeat=self.n-1))

    def _recursive_smaller_ngram(self):
        """
        Recursively generates (n-1)-gram models down to 1-grams, based on the marginal probabilities of the n-gram model.
        """
        ngrams = {self.n: self}

        for k in range(self.n-1, 0, -1): # Generate k-gram based on the marginal distribution of k+1-gram
            prob_table = defaultdict(dict)
            for context, dist in ngrams[k+1].prob_table.items():
                marginal_context = context[1:]
                for token, prob in dist.items():
                    if token in prob_table[marginal_context]:
                        prob_table[marginal_context][token] += prob
                    else:
                        prob_table[marginal_context][token] = prob

            for context, dist in prob_table.items(): # Normalize probabilities
                total_prob = sum(dist.values())
                for token in dist:
                    prob_table[context][token] /= total_prob

            assert set(product(self.vocab, repeat=k-1)) == set(prob_table.keys()), "Make sure we don't miss any context"

            ngrams[k] = NGram(self.vocab, k, prob_table)

        return ngrams # reverse for ngrams from 1 to n
    
    def prob_dist(self, prev_tokens):
        '''Give the probability distribution for the next token given the previous tokens. If the context is smaller than n-1, 
        use the model marginalized over the first tokens.'''
        if len(prev_tokens) < self.n-1:
            return self.smaller_ngrams[len(prev_tokens)+1].prob_dist(prev_tokens)
        context = tuple(prev_tokens[-(self.n-1):])
        return self.prob_table[context]

    def generate(self, prev_tokens=None, num_tokens=1):
        generated = []
        if not prev_tokens:
            prev_tokens = []
        for _ in range(num_tokens):
            dist = self.prob_dist(prev_tokens)
            next_token = random.choices(list(dist.keys()), weights=list(dist.values()))[0]
            generated.append(next_token)
            prev_tokens = prev_tokens + [next_token]
        return generated
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def save_tokenized_seqs(self, filename, num_seqs, seq_length):
        seqs = [[self.vocab.index(t) for t in self.generate(num_tokens=seq_length)]
                        for _ in tqdm(range(num_seqs), desc='Creating sequences')]
        np.save(filename, np.array(seqs))

    def save_token_map(self, filename):
        token_map = np.array(self.vocab)
        np.savez(filename, token_map)


if __name__ == '__main__':
    vocab_size = 10
    n = 6
    seq_length = 10
    num_train_seqs = int(1e8)
    num_val_seqs = int(0.2*num_train_seqs)
    vocab = [str(i) for i in range(vocab_size)]

    print(f'Builiding NGrams with vocab_size = {vocab_size} and n <= {n}')
    ngram = NGram(vocab, n)

    print('Testing generation')
    inp = random.choice(vocab)
    print(f'{inp}... -> {ngram.generate([inp], seq_length)}')

    ngram.save('ngram/ngram.pkl')
    print('Saving train and validation datasets...')
    ngram.save_tokenized_seqs('ngram/train_seqs.npy', num_train_seqs, seq_length)
    ngram.save_tokenized_seqs('ngram/val_seqs.npy', num_val_seqs, seq_length)
    ngram.save_token_map('ngram/token_map.npz')
    print('Done!')
