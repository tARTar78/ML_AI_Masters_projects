import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calc_recall(true_labels, pred_labels, k, exclude_self=False, return_mistakes=False):
    '''
    счиатет recall@k для приближенного поиска соседей
    
    true_labels: np.array (n_samples, k)
    pred_labels: np.array (n_samples, k)
    
    exclude_self: bool
        Если query_data была в трейне, считаем recall по k ближайшим соседям, не считая самого себя
    return_mistakes: bool
        Возвращать ли ошибки
    
    returns:
        recall@k
        mistakes: np.array (n_samples, ) с количеством ошибок
    '''
    n = true_labels.shape[0]
    n_success = []
    shift = int(exclude_self)
    
    for i in range(n):
        n_success.append(np.intersect1d(true_labels[i, shift:k+shift], pred_labels[i, shift:k+shift]).shape[0])
        
    recall = sum(n_success) / n / k
    if return_mistakes:
        mistakes = k - np.array(n_success)
        return recall, mistakes
    return recall


def plot_ann_performance(build_data,query_data,index_dict,k,flat_build_func,flat_search_func,query_in_train=False,title = 'LOL',qps_line = 1e-1,recall_line=7e-1, **kwargs):
	btl = []
	mosaic = '''
	ABB
	.BB
	'''
	fig, ax = plt.subplot_mosaic(mosaic, figsize=(15, 8))
	indexf = flat_build_func(build_data)
	distancesf, labelsf, fst = flat_search_func(indexf, query_data, k)
	n_queries = query_data.shape[0]
	fst = n_queries / fst
	dst = []
	lbl = []
	#names = []
	for key, dicta in index_dict.items():
		#print(key)
		index, bt = dicta['build_func'](build_data,**(dicta['fixed_params']))
		#indexf = flat_build_func(build_data)
		#names.append()
		btl.append(bt)
		stl = []
		rcll = []
		for val in dicta['search_param'][1]:
			#print(val)
			s_p = {dicta['search_param'][0] : val}
			#print(val)
			distances, labels, st = dicta['search_func'](index, query_data, k, **s_p)
			#dst.append(distances)
			#lbl.append(labels)
			stl.append(n_queries / st)
			#distancesf, labelsf = flat_search_func(indexf, query_data.copy(), k)
			#print(labels)
			#print(labelsf)
			rcl = calc_recall(true_labels=labelsf,pred_labels=labels,k=k,exclude_self= query_in_train)
			rcll.append(rcl)
		ax['B'].plot(rcll,stl, marker='o', linestyle='-',label=key)
		for i in range(len(stl)):
		    txt = dicta['search_param'][0] + "=" + str(dicta['search_param'][1][i])
		    ax['B'].annotate(txt, (rcll[i], stl[i]))
		    #print(np.array(index_dict.keys()))
		    #print(np.array(btl))
	ke = np.array(list(index_dict.keys()))
	btl = np.array(btl)
	#print(ke)
	#print(btl)
	sns.barplot(x=ke, y=btl, ax=ax['A'])
	ax['A'].set_xticklabels(ke, rotation=90)
	ax['A'].set_title('Build time')
	ax['A'].grid ( True )
	ax['B'].grid ( True )
	ax['B'].set_yscale('log')
	#axs[0].set_xlabel('')
	ax['A'].set_ylabel('time,sec')
	ax['B'].axhline(y=qps_line, color="b", linestyle="--")  
	s = "flat: "+str("{:.3e}".format(fst))+" qps"
	ax['B'].axhline(y=fst, color="r", linestyle="--",label= s) 
	ax['B'].axvline(x=recall_line, color="b", linestyle="--")  
	ax['B'].legend()
	ax['B'].set_title(title)
	ax['B'].set_xlabel('recall@' + str(k))
	ax['B'].set_ylabel('queries per second')
	plt.show()
	return 0
    
def analyze_ann_method(build_data,query_data,build_func,search_func,k,flat_build_func,flat_search_func,query_in_train=False,index_name='some hnsw l2-index'):
    '''
    some docstring :)
    '''
    #fig, ax = plt.subplot_mosaic(figsize=(15, 8))
    miss =[]
    indexf = flat_build_func(build_data)
    distancesf, labelsf, fst = flat_search_func(indexf, query_data, k=k)
    index, bt = build_func(build_data,k=k)
    distances, labels, st = search_func(index,query_data,k=k)
    n_queries = query_data.shape[0]
    st = n_queries / st
    print(query_in_train)
    rcl,mis = calc_recall(true_labels=labelsf,pred_labels=labels,k=k,exclude_self=query_in_train,return_mistakes=True)
    mis = np.array(mis)
    unique_values, counts = np.unique(mis, return_counts=True)
    lk = np.arange(k+1)
    ly = np.zeros(k+1)
    ly[unique_values] = counts
    mxl = np.max(ly)
    ly[ly == 0] = -0.1*mxl
    #mis = np.sum(mis)
    #print(mis)
    #miss.append(mis)
    #miss = np.array(miss)
    fig, ax = plt.subplots(figsize=(10, 6))
    s1 = "build time: " + str("{:.3}".format(bt)) +",sec"
    s2 = "qps: " + str("{:.3e}".format(st))
    s3 ="recall@" + str(k) + ": " + str("{:.3}".format(rcl))
    ss = (s1+"\n"+s2+"\n"+s3)
    barc = ax.bar(lk,ly, label=ss )
    hs = []
    for p in barc.patches:
    	h = p.get_height()
    	hs.append(int(h))
    hs = np.array(hs)
    hs[hs < 0] = 0
    ax.bar_label(barc,labels = hs, label_type='edge')
    ax.set_xlabel('mistakes')
    ax.set_ylabel('count')
    s = index_name + "#mistakes per query of " + str(k) +"NN"
    s1 = "build time: " + str(bt) +",sec"
    s2 = "qps: " + str("{:.3e}".format(st))
    s3 ="recall@" + str(k) + ": " + str(rcl)
    #ss = (s1+"\n"+s2+"\n"+s3)
    ax.set_title(s)
    ax.legend()
    plt.show()
    return 0



# Для FASHION MNIST
def knn_predict_classification(neighbor_ids, tr_labels, n_classes, distances=None, weights='uniform'):
    '''
    по расстояниям и айдишникам получает ответ для задачи классификации
    
    distances: (n_samples, k) - расстояния до соседей
    neighbor_ids: (n_samples, k) - айдишники соседей
    tr_labels: (n_samples,) - метки трейна
    n_classes: кол-во классов
     
    returns:
        labels: (n_samples,) - предсказанные метки
    '''
    
    n, k = neighbor_ids.shape

    labels = np.take(tr_labels, neighbor_ids)
    labels = np.add(labels, np.arange(n).reshape(-1, 1) * n_classes, out=labels)

    if weights == 'uniform':
        w = np.ones(n * k)
    elif weights == 'distance' and distances is not None:
        w = 1. / (distances.ravel() + 1e-10)
    else:
        raise NotImplementedError()
        
    labels = np.bincount(labels.ravel(), weights=w, minlength=n * n_classes)
    labels = labels.reshape(n, n_classes).argmax(axis=1).ravel()
    return labels


# Для крабов!
def get_k_neighbors(distances, k):
    '''
    считает по матрице попарных расстояний метки k ближайших соседей
    
    distances: (n_queries, n_samples)
    k: кол-во соседей
    
    returns:
        labels: (n_queries, k) - метки соседей
    '''
    indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    lowest_distances = np.take_along_axis(distances, indices, axis=1)
    neighbors_idx = lowest_distances.argsort(axis=1)
    indices = np.take_along_axis(indices, neighbors_idx, axis=1) # sorted
    sorted_distances = np.take_along_axis(distances, indices, axis=1)
    return sorted_distances, indices


# Для крабов! Пишите сами...
#def knn_predict_regression(labels, y, weights='uniform', distances=None):
#    '''
#    по расстояниям и айдишникам получает ответ для задачи регрессии
#    '''
#    if weights == 'uniform':
#        predicted_value = np.mean(y[labels], axis=1)
#        return predicted_value
#    elif weights == 'distance':
#        if distances is not None:
#            weights = 1 / (distances + 1e-6)  # добавляется небольшое число для стабильности
#            weighted_targets = y[labels] * weights
#            predicted_value = np.sum(weighted_targets,axis=1) / np.sum(weights,axis=1)
#            return predicted_value
#        else:
#            raise ValueError("Distances")
            
def knn_predict_regression(labels, y, weights='uniform', distances=None):
    '''
    по расстояниям и айдишникам получает ответ для задачи регрессии
    '''
    
    labels = np.take(y, labels)

    if weights == 'uniform':
        w = np.ones_like(labels)
    elif weights == 'distance' and distances is not None:
        w = 1. / (distances + 1e-10)
    else:
        raise NotImplementedError()
    
    pred = (labels * w).sum(axis=1) / w.sum(axis=1)
    return pred
