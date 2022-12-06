import copy
from random import choices
from code_utils.set_noise import *



def compute_grad_update(old_model, new_model, device=None):
	# maybe later to implement on selected layers/parameters
	if device:
		old_model, new_model = old_model.to(device), new_model.to(device)
	return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]



class Client(object):

	def __init__(self, conf, model, train_dataset_all,data_indices,test_dataset_few, id ):

		self.conf = conf

		self.local_model = model

		self.client_id = id

		self.train_dataset = train_dataset_all

		self.test_dataset = test_dataset_few

		all_range = list(range(len(self.train_dataset)))

		if self.conf["no_iid"] == "no":
			if self.conf["dataset"] == "adult":
				data_indice = choices(data_indices[0], k=2500) + choices(data_indices[1], k=2500)

			elif self.conf["dataset"] == "bank":
				data_indice = choices(data_indices[0], k=1000) + choices(data_indices[1], k=1000)

			elif self.conf["dataset"] == "gtrsb":
				data_indice = []
				for i in range(43):
					data_indice += choices(data_indices[i], k=120)
			elif self.conf["dataset"] == "mnist":
				data_indice = []
				for i in range(10):
					data_indice += choices(data_indices[i], k=600)

			elif self.conf["dataset"] == "cancer":
				data_indice = choices(data_indices[0], k=100) + choices(data_indices[1], k=100)


			else:
				data_len = int(len(self.train_dataset) / 10)
				data_indice = choices(all_range, k=data_len)





		if conf["no_iid"] == "yes":
			self.train_dataset = train_dataset_all
			data_indice = data_indices[id]







		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
								sampler=torch.utils.data.sampler.SubsetRandomSampler(data_indice))


		self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=conf["batch_size"])



	def local_train(self, model):

		local_model = copy.deepcopy(model).cuda()



		optimizer = torch.optim.SGD(local_model.parameters(), lr=self.conf['lr'],momentum=self.conf['momentum'])


		local_model.train()


		for e in range(self.conf["local_epochs"]):

			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch

				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()

				optimizer.zero_grad()
				output = local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target.long())
				loss.backward()


				optimizer.step()

		diff = dict()
		local_parameter = dict()

		for name, data in local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			local_parameter[name] = data


		return diff


	def verification(self, local_model):

		local_model.eval()
		correct = 0
		dataset_size = 0
		with torch.no_grad():
			for e in range(self.conf["local_epochs"]):

				for batch_id, batch in enumerate(self.test_loader):
					data, target = batch
					dataset_size += data.size()[0]

					if torch.cuda.is_available():
						data = data.cuda()
						target = target.cuda()


					output = local_model(data)

					pred = output.data.max(1)[1]  # get the index of the max log-probability
					correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		return acc



