import argparse, json
import datetime
import os
import logging
import torch,random
import time
from server import *
from client import *

from code_utils.dirichlet import *
from code_utils.datasets import *
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = "cuda" if torch.cuda.is_available() else "cpu"



if __name__ == '__main__':



	torch.manual_seed(1234)
	np.random.seed(1234)
	random.seed(1234)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf',default="./code_utils/conf.json" ,dest='conf')
	# parser.add_argument('--type',type=str,default='mnist')
	# parser.add_argument('--model',type=str,default='LeNet')
	# parser.add_argument('--no_models',type=int,default=10)
	# parser.add_argument('--global_epochs',type=int,default=20)
	# parser.add_argument('--free_riders',type=int,default=1)
	# parser.add_argument('--k',type=int,default=5)
	# parser.add_argument('--local_epochs',type=int,default=3)
	# parser.add_argument('--batch_size',type=int,default=32)
	# parser.add_argument('--lr',type=float,default=0.001)
	# parser.add_argument('--momentum',type=float,default=0.0001)
	# parser.add_argument('--lambda',type=float,default=0.1)


	args = parser.parse_args()

	with open(args.conf, 'r') as f:
		conf = json.load(f)
	

	train_datasets_all, eval_datasets  ,test_dataset_few= get_dataset("./data/", conf["dataset"],device)
	server = Server(conf, eval_datasets,test_dataset_few)
	clients = []

	if conf["dataset"] == "gtrsb":
		dataset = load_dataset('./data/gtrsb/gtsrb_dataset.h5', keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
		Y_train = np.array(dataset['Y_train'], dtype='int64')
		Y_train = np.asarray([np.where(r == 1)[0][0] for r in Y_train])
	else:
		Y_train = 0

	if conf["no_iid"] == "yes":
		if conf["dataset"] == "adult" or conf["dataset"] == "bank":
			data_indices = partition_data(conf, Y_train, train_datasets_all, conf["k"], conf["beta"])
		else:
			data_indices = partition_data(conf, Y_train, train_datasets_all, 10, conf["beta"])



	else:
		data_indices = []
		if conf["dataset"] == "adult":
			for k in range(2):
				idx_k = np.where(train_datasets_all.targets.cpu().numpy() == k)[0]
				data_indices.append(idx_k.tolist())
		elif conf["dataset"] == "bank":
			for k in range(2):
				idx_k = np.where(train_datasets_all.y == k)[0]
				data_indices.append(idx_k.tolist())
		elif conf["dataset"] == "gtrsb":
			for k in range(43):
				idx_k = np.where(Y_train == k)[0]
				data_indices.append(idx_k.tolist())
		elif conf["dataset"] == "mnist":
			for k in range(10):
				idx_k = np.where(train_datasets_all.targets == k)[0]
				data_indices.append(idx_k.tolist())

		else:
			data_indices = 0




	for c in range(conf["k"]):

			clients.append(Client(conf, server.global_model, train_datasets_all, data_indices,test_dataset_few, c))

		
	print("\n\n")

	local_acc_final = []
	server_acc_final = []
	server_acc_2_final = []

	global_model_list = []



	candidates = []

	for j in range(conf["k"]+conf["free_riders"]):
		candidates.append(clients[j])



	av_count_all = []

	threshold_value_all = []

	A = []
	B = []
	cout = 0
	FLAG = 0


	for e in range(conf["global_epochs"]):
		start = time.time()
		w_locals = []
		weight_accumulator = {}
		acc_all = []
		local_Cosine_similarity_all = []
		weight_all = []
		weight_all1 = []

		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)

		for i,c in enumerate(candidates):

			diff = c.local_train(server.global_model)

			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])



		server.model_aggregate(weight_accumulator)

		acc_server, loss = server.model_eval(device)


		print(acc_server)