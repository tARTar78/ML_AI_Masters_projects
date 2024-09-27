from time import time as tm
import faiss
import hnswlib
import numpy as np

def timer(func):
    '''
    декоратор, замеряющий время работы функции
    '''
    def wrapper(*args, **kwargs):
        start_time = tm()
        result = func(*args, **kwargs)
        end_time = tm() - start_time
        if isinstance(result, tuple):
            return *result, end_time
        return result, end_time
    return wrapper


@timer
def build_IVFPQ(build_data, **fixed_params):
    dim = fixed_params['dim']
    coarse_index = fixed_params['coarse_index']
    nlist = fixed_params['nlist']
    m = fixed_params['m']
    nbits = fixed_params['nbits']
    metric = fixed_params['metric']
    
    num_threads = fixed_params.get('num_threads', 1)
    faiss.omp_set_num_threads(num_threads)
    
    index = faiss.IndexIVFPQ( # у faiss туго с именованными аргументами
        coarse_index, # индекс для поиска соседей-центроидов
        dim, # размерность исходных векторов
        nlist, # количество coarse-центроидов = ячеек таблицы
        m, # на какое кол-во подвекторов бить исходные для PQ
        nbits, # log2 k* - количество бит на один маленький (составной) PQ-центроид
        metric # метрика, по которой считается расстояние между остатком(q) и [pq-центроидом остатка](x)
    )
    index.train(build_data)
    index.add(build_data)
    return index # из-за декоратора ожидайте, что возвращается index, build_time

@timer
def build_hnsw(build_data, **fixed_params):
    dim = fixed_params['dim']
    space = fixed_params['space']
    M = fixed_params['M']
    ef_construction = fixed_params['ef_construction']
    #print(dim)
    #print(space)
    #print(M)
    #print(ef_construction)
    p = hnswlib.Index(space=space, dim=dim) # possible options are l2, cosine or ip

	# Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=build_data.shape[0], ef_construction=ef_construction, M=M)
    #print("AA")
    ids = np.arange(build_data.shape[0])
    p.add_items(data=build_data,ids=ids)
    
    return p # из-за декоратора ожидайте, что возвращается index, build_time

@timer
def search_hnsw(index, query_data, k, efSearch=10):
    #print(efSearch)
    index.set_ef(efSearch)
    labels, distances = index.knn_query(data=query_data, k=k)
    return distances, labels # из-за декоратора ожидайте, что возвращается distances, labels, search_time

def build_flat_l2(build_data, dim):
    index = faiss.IndexFlatL2(dim)
    index.add(build_data)
    return index
    
def build_flat_ip(build_data, dim):
    index = faiss.IndexFlatIP(dim)
    index.add(build_data)
    return index
    
def build_flat_cos(build_data, dim):
    vec = build_data.copy()
    faiss.normalize_L2(vec)
    index = faiss.IndexFlatIP(dim)
    index.add(vec)
    return index

@timer
def search_flat(index, query_data, k):
    distances, labels = index.search(query_data, k)
    return distances, labels

@timer
def build_IVF_cos(build_data, **fixed_params):
    dim = fixed_params['dim']
    coarse_index = fixed_params['coarse_index']
    nlist = fixed_params['nlist']
    #m = fixed_params['m']
    #nbits = fixed_params['nbits']
    metric = fixed_params['metric']
    num_threads = fixed_params['num_threads']
    
    num_threads = fixed_params.get('num_threads', 1)
    faiss.omp_set_num_threads(num_threads)
    #print(coarse_index)
    #print(dim)
    #print(nlist)
    #print(faiss.METRIC_L2)
    #print(metric)
    index = faiss.IndexIVFFlat( # у faiss туго с именованными аргументами
        coarse_index, # индекс для поиска соседей-центроидов
        dim, # размерность исходных векторов
        nlist, # количество coarse-центроидов = ячеек таблицы # на какое кол-во подвекторов бить исходные для PQ # log2 k* - количество бит на один маленький (составной) PQ-центроид
        metric # метрика, по которой считается расстояние между остатком(q) и [pq-центроидом остатка](x)
    )
    vec = build_data.copy()
    faiss.normalize_L2(vec)
    index.train(vec)
    index.add(vec)
    return index # из-за декоратора ожидайте, что возвращается index, build_time



@timer
def build_IVF(build_data, **fixed_params):
    dim = fixed_params['dim']
    coarse_index = fixed_params['coarse_index']
    nlist = fixed_params['nlist']
    #m = fixed_params['m']
    #nbits = fixed_params['nbits']
    metric = fixed_params['metric']
    num_threads = fixed_params['num_threads']
    
    num_threads = fixed_params.get('num_threads', 1)
    faiss.omp_set_num_threads(num_threads)
    #print(coarse_index)
    #print(dim)
    #print(nlist)
    #print(faiss.METRIC_L2)
    #print(metric)
    index = faiss.IndexIVFFlat( # у faiss туго с именованными аргументами
        coarse_index, # индекс для поиска соседей-центроидов
        dim, # размерность исходных векторов
        nlist, # количество coarse-центроидов = ячеек таблицы # на какое кол-во подвекторов бить исходные для PQ # log2 k* - количество бит на один маленький (составной) PQ-центроид
        metric # метрика, по которой считается расстояние между остатком(q) и [pq-центроидом остатка](x)
    )
    index.train(build_data)
    index.add(build_data)
    return index # из-за декоратора ожидайте, что возвращается index, build_time


@timer
def search_faiss(index, query_data, k, n_probe=1):
    index.nprobe = n_probe # количество ячеек таблицы, в которые мы заглядываем. Мы заглядываем в nprobe ближайших coarse-центроидов для q
    distances, labels = index.search(query_data, k)
    return distances, labels # из-за декоратора ожидайте, что возвращается distances, labels, search_time
