import PyEinsummable as pe
import numpy as np
import torch, time

from datasets import load_dataset
from transformers import LlamaTokenizer


def main():
    weights = torch.load("../../mnt/Llama1/consolidated.00.pth", weights_only = True)
    for (name, weight) in weights.items():
        if name.find("rope") == -1 and name.find("tok") == -1:
            if torch.isnan(weight).any():
                print(name + " has nan values")



    embedding_matrix = weights["tok_embeddings.weight"]
    
    dbuf_weights = load_llama_weights(weights)

    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = 4096
    tokenizer.pad_token = tokenizer.unk_token
    
    model = pe.llama(125 * 100 * 1000 * 1000, int(35e10), pe.llama_size.B7)
    print("created model")
        
    print("loading tensors ...")
    model.load_tensors(dbuf_weights)
    print("tensors loaded")
    
    ds = load_dataset("garage-bAInd/Open-Platypus")
    
    combined_ds = ds.map(lambda x: {"full_input" : x['instruction'] + x["output"]})

    def tokenize_function(examples):
        return tokenizer(examples["full_input"], padding='max_length', truncation=True, return_token_type_ids=False)

    tokenized_datasets = combined_ds.map(tokenize_function, batched=True)

    tokens = [torch.tensor(tokenized_datasets['train']['input_ids'][i]) for i in range(6)]

    batched_tokens = []
    batched_labels = []
        
    for token_list in tokens:
        batch_of_tokens, batch_of_labels = mask_split(token_list, tokenizer.pad_token_id)
        temp = []
        for tok in batch_of_tokens:
            temp.append(dbuf_from_numpy(embedding_matrix[torch.reshape(tok, (1, 4096))].to(torch.float).numpy()))
        batched_tokens.append(temp)
        temp = []
        for label in batch_of_labels:
            temp.append(dbuf_from_numpy(label.to(torch.float).numpy()))
        batched_labels.append(temp)
    
    
    # batched_tokens = [dbuf_from_numpy(embedding_matrix[torch.reshape(torch.tensor(tokenized_datasets['train']['input_ids'][i]), (1, 4096))].to(torch.float).numpy()) for i in range(6)]

    # batched_tokens_torch = [embedding_matrix[torch.randint(0, 32000, (1, 4096))] for _ in range(10)]
    # for tok in batched_tokens_torch:
    #     assert(not torch.isnan(tok).any())

    # batched_tokens = [dbuf_from_numpy(tok.to(torch.float).numpy()) for tok in batched_tokens_torch]

    # for (name, weight) in weights.items():
    #     if name.find("rope") == -1 and name.find("tok") == -1:
    #         dbuf = model.get_tensor(name)
    #         if (abs(float(dbuf.sum().str()) - weight.to(torch.float).sum().item()) > 1.0):
    #             print(name + " does not match loaded tensor, sums are: einsum: " + dbuf.sum().str() + ", torch: " + str((weight.to(torch.float).sum().item())))

    print("Assertions complete, starting to train")    
    model.train(3, batched_tokens, batched_labels)

def mask_split(input_tokens, pad_token_id):

  idx = 0
  batched_tokens = []
  batched_labels = []

  while input_tokens[idx] != pad_token_id and idx < len(input_tokens):
    mask = torch.tensor([1 if i < idx else 0 for i in range(len(input_tokens))])
    batched_tokens.append(input_tokens * mask)
    one_hot = torch.zeros(32000)
    one_hot[input_tokens[idx+1]] = 1
    batched_labels.append(one_hot)
    idx += 1
  return batched_tokens, batched_labels


def test_dbufs():
    # Uncomment to see class overviews
    # help(PyEinsummable.server)
    # help(PyEinsummable.array)
    
    memory = int(10e6)
    storage = int(10e6)
    eps = 10e-3
    np.random.seed(1234)
    
    input1 = np.random.normal(size=(200, 200)).astype(np.float32)
    #input1 = np.arange(40000).reshape((200, 200)).astype(np.float32)
    #input1 = np.random.uniform(size=(200, 200)).astype(np.float32)
    input1_np = np.copy(input1)
    input1_dbuf = dbuf_from_numpy(input1)
    # Sanity check values
    for i in range(25000, 25100):
        assert(input1[i // 200][i - ((i // 200) * 200)] - pe.scalar_to_float(input1_dbuf.get(i)) < eps)
    
    # np.repeat(np.pad(np.arange(20), (0, 180)), repeats=200).reshape((200, 200))
    # np.repeat([0, 1], repeats=20000).reshape((200, 200))
    
    
    input2 = np.random.normal(size=(200, 200)).astype(np.float32)
    #input2 = np.arange(40000).reshape((200, 200)).astype(np.float32)
    #input2 = np.random.uniform(size=(200, 200)).astype(np.float32)
    input2_np = np.copy(input2)
    input2_dbuf = dbuf_from_numpy(input2)
    # Sanity check values
    for i in range(33000, 33100):
        assert(input2[i // 200][i - ((i // 200) * 200)] - pe.scalar_to_float(input2_dbuf.get(i)) < eps)
    
    # Start and shutdown server
    serv = pe.server([memory], storage)
    print("Successfully created server")
    
    gc = pe.matmul_graph_con(1, 1, 1, 200, 200, 200, 1, memory)
    
    matmul_graph = gc.graph
    matmul_graph.print()
    
    serv.insert_tensor(0, [200, 200], input1_dbuf)
    serv.insert_tensor(1, [200, 200], input2_dbuf)
    # serv.insert_gid_without_data(3, pe.relation.singleton(pe.dtype.f32, [200, 200], 3))
    
    placements = [0, 0, 0, 0]
    for k in gc.placements:
        placements[k] = gc.placements[k]
    
    serv.execute_graph(gc.graph, placements, {})
      
    out_np = np.einsum("ij,jh", input1_np, input2_np);
    #out_np = np.matmul(input1_np, input2_np)
    print("Numpy first result: ", out_np[0][0])
    
    # rel = serv.get_relation(2)
    out_dbuf = serv.get_tensor(3);
    #pe.print_dbuf(out_dbuf)
    out_as_np = np.frombuffer(out_dbuf, dtype=np.float32).reshape((200, 200))
    print("Einsummable first result: ", out_dbuf.get(0).str());
    
    print("Numpy sum: ", np.sum(out_np))
    print("Einsummable sum: ", out_dbuf.sumf64())
    
    assert(np.isclose(out_np, out_as_np).all())
    
    serv.shutdown()
    print("Server shutdown")
    
    
    test_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.half)
    buf = dbuf_from_numpy(test_array)
    print(buf.get(2).str())
    

def load_llama_weights(model_dict: dict):
    out_dict = dict()
    
    start = time.time()
    for name, weight in model_dict.items():
        weight = weight.to(torch.float).numpy()
        # print(name + " has shape " + str(weight.shape))
        out_dict[name] = dbuf_from_numpy(weight)
        
    print(f"Changing dtype and copying to dbuffer_ts took {time.time()-start}s")
    
    return out_dict
    
def dbuf_from_numpy(array):
    typestr = str(array.__array_interface__['typestr'])

    if typestr.startswith("|") or typestr.startswith(">"):
        raise ValueError("Byte-ordering not supported")
    if typestr[1] not in ["i", "u", "f", "c"]:
        raise ValueError("Array type not implemented")
    num_bytes = int(typestr[2:])
    if num_bytes not in [2, 4, 8] or (typestr[1] == "c" and num_bytes != 8):
        raise ValueError("Size of values not supported")
    
    if typestr[1] in ["i", "u"]:
        typestr = typestr[:1] + "f" + typestr[2:]
        array = array.astype(typestr)

    dtype = (typestr[1], num_bytes)
    if dtype == ("c", 8):
        return pe.dbuf_from_numpy(array.__array_interface__["data"][0], np.prod(array.shape), pe.dtype.c64)
    elif dtype == ("f", 2):
        return pe.dbuf_from_numpy(array.__array_interface__["data"][0], np.prod(array.shape), pe.dtype.f16)
    elif dtype == ("f", 4):
        return pe.dbuf_from_numpy(array.__array_interface__["data"][0], np.prod(array.shape), pe.dtype.f32)
    elif dtype == ("f", 8):
        return pe.dbuf_from_numpy(array.__array_interface__["data"][0], np.prod(array.shape), pe.dtype.f64)
    
    
    
    
        
     
    
    
if __name__ == "__main__":
    main()