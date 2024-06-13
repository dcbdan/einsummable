import PyEinsummable as pe
import numpy as np
import torch, time


def main():
    # Uncomment to see class overviews
    # help(PyEinsummable.server)
    # help(PyEinsummable.array)
    
    memory = int(10e6)
    storage = int(10e6)
    
    input1 = np.random.normal(size=(200, 200)).astype(np.float32)
    print(input1.__array_interface__['typestr'][2:])
    input1_dbuf = dbuf_from_numpy(input1)
    input2 = np.random.normal(size=(200, 200)).astype(np.float32)
    input2_dbuf = dbuf_from_numpy(input1)
    
    # Start and shutdown server
    serv = pe.server([memory], storage)
    print("Successfully created server")
    
    gc = pe.matmul_graph_con(2, 2, 2, 100, 100, 100, 1, memory)
    
    matmul_graph = gc.graph
    matmul_graph.print()
    
    serv.insert_tensor(0, [200, 200], input1_dbuf)
    serv.insert_tensor(1, [200, 200], input2_dbuf)
    # serv.insert_gid_without_data(3, pe.relation.singleton(pe.dtype.f32, [200, 200], 3))
    
    placements = [0, 0, 0, 0]
    for k in gc.placements:
        placements[k] = gc.placements[k]
    
    serv.execute_graph(gc.graph, placements, {})
      
    out_np = np.matmul(input1, input2);
    print("Numpy first result: ", out_np[0][0])
    
    # rel = serv.get_relation(2)
    out_dbuf = serv.get_tensor(3);
    out_numpy = np.frombuffer(out_dbuf, dtype=np.float32).reshape((200, 200))
    print(out_np)
    print(out_numpy)
    print("Einsummable first result: ", out_dbuf.get(0).str());
    
    serv.shutdown()
    print("Server shutdown")
    
    
    test_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.half)
    buf = dbuf_from_numpy(test_array)
    print(buf.get(2).str())
    


def load_llama_weights(filename: str):
    start = time.time()
    model_dict = torch.load(filename)
    print(f"PyTorch.load() for Llama3-8B took {time.time()-start}s")
    
    
    start = time.time()
    for weight in model_dict.values():
        weight = weight.to(torch.half).numpy()
        dbuf = dbuf_from_numpy(weight)
    print(f"Changing dtype and making dbuffer_ts for Llama3-8B took {time.time()-start}s")
    
    print(dbuf.get(2).str())
    
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
        typestr[1] = "f"
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