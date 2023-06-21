import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import statistics
import sqlite3
import numpy as np


bert_model = "bert-base-uncased"
tokenizer_model = bert_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sqlite_file = "partial.sqlite"
learning_rate = 1e-5
batch_size = 32
epochs = 5
maxlen = 256
loss_fn = nn.BCEWithLogitsLoss().to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)


def one_hot(index, length):
	vector = np.zeros(length)
	vector[index] = 1
	return vector


class CustomDataset(Dataset):

	def __init__(self, sqlite_file, maxlen, mode):
		self.cur = sqlite3.connect(sqlite_file).cursor()
		self.maxlen = maxlen
		self.mode = mode
		self.size = self.cur.execute("SELECT count(*) FROM pages").fetchone()[0]-1
		self.validate_ids = random.sample([i for i in range(self.size)], 50)

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		index = random.randint(1, self.size)
		self.cur.execute("SELECT content, category_id FROM pages WHERE count = ?", (index, ))
		row = self.cur.fetchone()
		content, category_id = row
		return content, one_hot(category_id, 7)


class SentencePairClassifierDisambiguationLarge(nn.Module):

	def __init__(self, bert_model, freeze_bert=False):
		super(SentencePairClassifierDisambiguationLarge, self).__init__()
		self.bert_layer = AutoModel.from_pretrained(bert_model, return_dict=False)
		self.cls_layer = nn.Linear(768, 7)
		self.dropout = nn.Dropout(p=0.1)
		if freeze_bert:
			for p in self.bert_layer.parameters():
				p.requires_grad = False

	def forward(self, input_ids, attn_masks, token_type_ids):
		_, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)
		logits = self.cls_layer(self.dropout(pooler_output))
		return logits


def train_loop(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	for batch, (sent, category_id) in enumerate(dataloader):
		encoded_pair = tokenizer(sent, None, padding='max_length', truncation=True, max_length=maxlen, return_tensors='pt')
		token_ids = encoded_pair['input_ids']
		attn_masks = encoded_pair['attention_mask']
		token_type_ids = encoded_pair['token_type_ids']
		pred = model(token_ids.to(device), attn_masks.to(device), token_type_ids.to(device))
		loss = loss_fn(pred, category_id.to(device))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		loss, current = loss.item(), batch
		print(f"loss: {loss:>7f}  [{current:>5d}/{size/batch_size}]")


def do_train():
	model = SentencePairClassifierDisambiguationLarge(bert_model, freeze_bert=False).train().to(device)
	train_set = CustomDataset(sqlite_file, maxlen, "train")
	train_loader = DataLoader(train_set, batch_size=batch_size)
	optimizer = torch.optim.Adam(params = model.parameters(), lr=learning_rate)
	for epoch in range(epochs):
		train_loop(train_loader, model, loss_fn, optimizer)
	torch.save(model.state_dict(), "model.pth")


def main():
	sent1 = "Chipco is a non listed company that sells chips"
	model = SentencePairClassifierDisambiguationLarge(bert_model, freeze_bert=False).train().to(device)
	model.load_state_dict(torch.load("model.pth"))
	model.eval()
	with torch.no_grad():
		encoded_pair = tokenizer(sent1, None, padding='max_length', truncation=True, max_length=maxlen, return_tensors='pt')
		token_ids = encoded_pair['input_ids']
		attn_masks = encoded_pair['attention_mask']
		token_type_ids = encoded_pair['token_type_ids']
		logits = model(token_ids.to(device), attn_masks.to(device), token_type_ids.to(device))
		sigmoid = torch.nn.Sigmoid()
		probs = sigmoid(logits)
		labels = probs.cpu().numpy()[0]
		print(labels)


if __name__ == "__main__":
	main()


# model.train()
# for t in range(epochs):
#   print(f"Epoch {t+1}\n-------------------------------")
#   train_loop(train_loader, model, loss_fn, optimizer)

# print("Done!")


# model.eval()
# cur = sqlite3.connect(sqlite_file).cursor()


#sent1 = "Chipco is a non listed company that sells chips"
#sent1 = '''Bridge Over Troubled Water is the fifth and final studio album by American folk rock duo Simon & Garfunkel, released in January 1970 on Columbia Records. Following the duo's soundtrack for The Graduate, Art Garfunkel took an acting role in the film Catch-22, while Paul Simon worked on the songs, writing all tracks except Felice and Boudleaux Bryant's "Bye Bye Love" (previously a hit for the Everly Brothers). '''
# sent1 = '''Frankfurt Airport (IATA: FRA, ICAO: EDDF; German: Flughafen Frankfurt Main [ˈfluːkhaːfn̩ ˈfʁaŋkfʊʁt ˈmaɪn], also known as Rhein-Main-Flughafen) is Germany's main international airport by passenger numbers[5] and is located in Frankfurt, the fifth-largest city of Germany and one of the world's leading financial centres. It is operated by Fraport and serves as the main hub for Lufthansa, including Lufthansa CityLine and Lufthansa Cargo as well as Condor and AeroLogic. The airport covers an area of 2,300 hectares (5,683 acres) of land[6] and features two passenger terminals with capacity for approximately 65 million passengers per year; four runways; and extensive logistics and maintenance facilities. '''
#sent1 = "An airport is a place where airplanes land and take off and fly around."
#sent1 = "Lake Titicaca (/tɪtɪˈkɑːkə/;[4] Spanish: Lago Titicaca [ˈlaɣo titiˈkaka]; Quechua: Titiqaqa Qucha) is a large freshwater lake in the Andes mountains on the border of Bolivia and Peru. It is often called the highest navigable lake in the world. By both volume of water and by surface area, it is the largest lake in South America."


# with torch.no_grad():

#     encoded_pair = tokenizer(sent1, None,
#                                     padding='max_length',  # Pad to max_length
#                                     truncation=True,  # Truncate to max_length
#                                     max_length=maxlen,
#                                     return_tensors='pt')  # Return torch.Tensor objects

#     token_ids = encoded_pair['input_ids']  # tensor of token ids
#     attn_masks = encoded_pair['attention_mask']  # binary tensor with "0" for padded values and "1" for the other values
#     token_type_ids = encoded_pair['token_type_ids']  # binary tensor with "0" for padded values and "1" for the other values


#     logits = model(token_ids.to(device), attn_masks.to(device), token_type_ids.to(device))
#     print(logits.shape)
#     sigmoid = torch.nn.Sigmoid()
#     probs = sigmoid(logits)
#     labels = probs.cpu().numpy()[0]

#     select_query = f"SELECT * FROM categories"
#     cur.execute(select_query)
#     result = cur.fetchall()

#     scored_labels = []
#     for i in range(len(labels)):
#       scored_labels.append([labels[i], result[i][1]])

#     scored_labels.sort(key = lambda x: x[0], reverse=True)

#     print(scored_labels)
