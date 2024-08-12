import PyEinsummable as pe
import numpy as np
import torch, time


def main():
    model = pe.llama(125 * 100 * 1000 * 1000, int(35e10), pe.llama_size.B7)
    
    weights = load_llama_weights("mnt/Llama1/consolidated.00.pth")
    
    model.load_tensors(weights)
    
    model.train(3)

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
    


def load_llama_weights(filename: str):
    start = time.time()
    model_dict = torch.load(filename)
    print(f"PyTorch.load() for Llama3-8B took {time.time()-start}s")
    out_dict = dict()
    
    start = time.time()
    for name, weight in model_dict.items():
        weight = weight.to(torch.half).numpy()
        out_dict[name] = dbuf_from_numpy(weight)
        
    print(f"Changing dtype and making dbuffer_ts for Llama3-8B took {time.time()-start}s")
    
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