import random
import numpy as np
import pickle
from collections import defaultdict
from itertools import product
from tqdm import tqdm

class NGramModel:
    def __init__(self, vocab, n):
        self.vocab = vocab
        self.n = n
        self.prob_table = self._initialize_prob_table()

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

    def prob_dist(self, prev_tokens):
        context = tuple(prev_tokens[-(self.n-1):])
        return self.prob_table[context]

    def generate(self, prev_tokens):
        probs = self.prob_dist(prev_tokens)
        return random.choices(list(probs.keys()), weights=list(probs.values()))[0]

class MultiNGram:
    def __init__(self, vocab, k):
        self.vocab = vocab # a list of tokens
        self.k = k # the maximum ngram size
        self.ngrams = [NGramModel(vocab, n) for n in range(1, k+1)]
        self.prob_lookup = self._build_prob_lookup()

    def _build_prob_lookup(self):
        lookup = {}
        total_combinations = sum(len(self.vocab) ** i for i in range(self.k))
        
        with tqdm(total=total_combinations, desc="Building probability lookup") as pbar:
            for length in range(self.k):
                for context in product(self.vocab, repeat=length):
                    lookup[context] = self._calculate_prob_dist(context)
                    pbar.update(1)
        
        return lookup

    def _calculate_prob_dist(self, prev_tokens):
        dist = defaultdict(float)
        num_models = min(len(prev_tokens) + 1, self.k)
        weight = 1 / num_models

        for n in range(1, num_models + 1):
            ngram_dist = self.ngrams[n-1].prob_dist(prev_tokens)
            for token, prob in ngram_dist.items():
                dist[token] += prob * weight
       
        # Normalize the distribution
        total = sum(dist.values())
        for token in dist:
            dist[token] /= total
        
        return dict(dist)

    def prob_dist(self, prev_tokens):
        context = tuple(prev_tokens[-(self.k-1):])
        return self.prob_lookup[context]

    def generate(self, prev_tokens=None, num_tokens=1):
        generated = []
        if not prev_tokens:
            prev_tokens = []
        for _ in range(num_tokens):
            context = tuple(prev_tokens[-(self.k-1):])
            dist = self.prob_lookup[context]
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
    context_size = 6
    seq_length = 10
    num_train_seqs = int(1e4)
    num_val_seqs = int(0.2*num_train_seqs)
    vocab = [str(i) for i in range(vocab_size)]

    print(f'Builiding NGramModel with vocab_size = {vocab_size} and context_size = {context_size}')
    multi_ngram = MultiNGram(vocab, context_size)

    print('Testing generation')
    inp = random.choice(vocab)
    print(f'{inp}... -> {multi_ngram.generate([inp], 10)}')

    multi_ngram.save('ngram/multi_ngram.pkl')
    print('Saving train and validation datasets...')
    multi_ngram.save_tokenized_seqs('ngram/train_seqs.npy', num_train_seqs, seq_length)
    multi_ngram.save_tokenized_seqs('ngram/val_seqs.npy', num_val_seqs, seq_length)
    multi_ngram.save_token_map('ngram/token_map.npz')
    print('Done!')
