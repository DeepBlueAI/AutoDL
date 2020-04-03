# -*- coding: utf-8 -*-
import gzip
import os
import time
import numpy as np
import gc


class GET_EMBEDDING:
    def __init__(self, language):
        stime = time.time()
        embedding_path = '/app/embedding'
        fasttext_embeddings_index = {}

        self.fasttext_embeddings_index_zh = {}
        self.fasttext_embeddings_index_en = {}
        
        if language == 'ZH':
            print ('load zh embedding...')
            f_zh = gzip.open(os.path.join(embedding_path, 'cc.zh.300.vec.gz'),'rb')
            
            for line in f_zh.readlines():               
                values = line.strip().split()
                word = values[0].decode('utf8')
                coefs = np.asarray(values[1:], dtype='float32')
                self.fasttext_embeddings_index_zh[word] = coefs
                #self.embedding_dict_zh[word] = coefs
                  
            #self.embedding_dict_zh = self.fasttext_embeddings_index_zh
            
            del f_zh, values, word, coefs
            gc.collect()
            print('read zh embedding time: {}s.'.format(time.time()-stime))
        else:
            print ('load en embedding...')
            f_en = gzip.open(os.path.join(embedding_path, 'cc.en.300.vec.gz'),'rb')
            for line in f_en.readlines():
                values = line.strip().split()
                word = values[0].decode('utf8')
                coefs = np.asarray(values[1:], dtype='float32')
                self.fasttext_embeddings_index_en[word] = coefs
                
                #self.embedding_dict_en[word] = coefs
            #self.embedding_dict_en = self.fasttext_embeddings_index_en
            
            del f_en, values, word, coefs
            gc.collect()
            print('read en embedding time: {}s.'.format(time.time()-stime))
        
        
