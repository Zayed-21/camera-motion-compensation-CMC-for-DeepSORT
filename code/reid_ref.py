"""Pure-numpy reference interpreter for the patched mars-small128 graph.
Used only to validate the OpenCV path (it is slow)."""
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from numpy.lib.stride_tricks import sliding_window_view as swv

f = descriptor_pb2.FileDescriptorProto(name="ref.proto", package="ref", syntax="proto3")
dim = f.message_type.add(name="Dim"); dim.field.add(name="size", number=1, type=3, label=1)
shp = f.message_type.add(name="Shape"); shp.field.add(name="dim", number=2, type=11, label=3, type_name=".ref.Dim")
tp = f.message_type.add(name="Tensor")
for nm, no, ty, lb in [("dtype",1,5,1),("tensor_content",4,12,1),("float_val",5,2,3),("int_val",7,5,3)]:
    tp.field.add(name=nm, number=no, type=ty, label=lb)
tp.field.add(name="tensor_shape", number=2, type=11, label=1, type_name=".ref.Shape")
lv = f.message_type.add(name="List"); lv.field.add(name="s", number=2, type=12, label=3); lv.field.add(name="i", number=3, type=3, label=3)
av = f.message_type.add(name="Attr")
for nm, no, ty in [("s",2,12),("i",3,3),("f",4,2),("b",5,8),("type",6,5)]:
    av.field.add(name=nm, number=no, type=ty, label=1)
av.field.add(name="list", number=1, type=11, label=1, type_name=".ref.List")
av.field.add(name="tensor", number=8, type=11, label=1, type_name=".ref.Tensor")
nd = f.message_type.add(name="Node")
for nm, no in [("name",1),("op",2)]: nd.field.add(name=nm, number=no, type=9, label=1)
nd.field.add(name="input", number=3, type=9, label=3)
e = nd.nested_type.add(name="AttrEntry"); e.options.map_entry = True
e.field.add(name="key", number=1, type=9, label=1); e.field.add(name="value", number=2, type=11, label=1, type_name=".ref.Attr")
nd.field.add(name="attr", number=5, type=11, label=3, type_name=".ref.Node.AttrEntry")
gd = f.message_type.add(name="Graph"); gd.field.add(name="node", number=1, type=11, label=3, type_name=".ref.Node")
pool = descriptor_pool.DescriptorPool(); pool.Add(f)
G = message_factory.GetMessageClass(pool.FindMessageTypeByName("ref.Graph"))
DT = {1: np.float32, 3: np.int32, 9: np.int64}

def const(n):
    t = n.attr["value"].tensor; dt = DT[t.dtype]
    shape = [d.size for d in t.tensor_shape.dim]
    if t.tensor_content: v = np.frombuffer(t.tensor_content, dt)
    else:
        v = np.array(t.float_val if dt == np.float32 else t.int_val, dt)
        if v.size == 1 and np.prod(shape) > 1: v = np.full(int(np.prod(shape)), v[0], dt)
    return v.reshape(shape) if shape else v.reshape(())

def pad_same(x, k, s):
    h, w = x.shape[1:3]
    ph = max((-(-h // s) - 1) * s + k - h, 0); pw = max((-(-w // s) - 1) * s + k - w, 0)
    return ph, pw

def conv(x, w, s, padding):
    k = w.shape[0]
    if padding == b"SAME":
        ph, pw = pad_same(x, k, s)
        x = np.pad(x, ((0,0),(ph//2, ph-ph//2),(pw//2, pw-pw//2),(0,0)))
    p = swv(x, (k, k), axis=(1, 2))[:, ::s, ::s]          # N,H',W',C,k,k
    return np.einsum("nhwcij,ijco->nhwo", p, w, optimize=True)

def maxpool(x, k, s, padding):
    if padding == b"SAME":
        ph, pw = pad_same(x, k, s)
        x = np.pad(x, ((0,0),(ph//2, ph-ph//2),(pw//2, pw-pw//2),(0,0)), constant_values=-np.inf)
    return swv(x, (k, k), axis=(1, 2))[:, ::s, ::s].max(axis=(-1, -2))

def run(pb, x, out="features"):
    g = G(); g.ParseFromString(open(pb, "rb").read())
    v = {}
    get = lambda name: v[name.split(":")[0]]
    for n in g.node:
        a = n.attr; i = [get(s) for s in n.input if not s.startswith("^")]
        op = n.op
        if op == "Placeholder": r = x.astype(np.float32)
        elif op == "Const": r = const(n)
        elif op == "Identity": r = i[0]
        elif op == "Conv2D":
            st = list(a["strides"].list.i); assert st[1] == st[2]
            r = conv(i[0], i[1], st[1], a["padding"].s)
        elif op == "MaxPool":
            k = list(a["ksize"].list.i)[1]; st = list(a["strides"].list.i)[1]
            r = maxpool(i[0], k, st, a["padding"].s)
        elif op == "FusedBatchNorm":
            xx, sc, b, m, var = i; eps = a["epsilon"].f
            r = (xx - m) / np.sqrt(var + eps) * sc + b
        elif op == "Elu": r = np.where(i[0] > 0, i[0], np.expm1(np.minimum(i[0], 0)))
        elif op in ("Add", "BiasAdd"): r = i[0] + i[1]
        elif op == "Shape": r = np.array(i[0].shape, np.int32)
        elif op == "StridedSlice":
            assert a["begin_mask"].i == 0 and a["shrink_axis_mask"].i == 1, "only x[b] supported"
            r = i[0][int(i[1][0])]
        elif op == "Pack": r = np.array([int(np.asarray(t).reshape(-1)[0]) for t in i], np.int32)
        elif op == "Reshape": r = i[0].reshape([int(s) for s in i[1]])
        elif op == "MatMul": r = i[0] @ i[1]
        elif op == "Square": r = i[0] ** 2
        elif op == "Sum": r = i[0].sum(axis=tuple(np.atleast_1d(i[1]).tolist()), keepdims=bool(a["keep_dims"].b))
        elif op == "Sqrt": r = np.sqrt(i[0])
        elif op == "RealDiv": r = i[0] / i[1]
        else: raise NotImplementedError(op)
        v[n.name] = r
        if n.name == out: return r
    return v[out]

class Tail:
    """Evaluate the graph from node `start` (given) to `features`, constants preloaded."""
    def __init__(self, pb, start):
        g = G(); g.ParseFromString(open(pb, "rb").read())
        names = [n.name for n in g.node]
        self.nodes = g.node[names.index(start) + 1:]
        self.start = start
        self.consts = {n.name: const(n) for n in g.node if n.op == "Const"}

    def __call__(self, x):
        v = dict(self.consts); v[self.start] = x
        get = lambda s: v[s.split(":")[0]]
        for n in self.nodes:
            if n.op == "Const": continue
            i = [get(s) for s in n.input if not s.startswith("^")]
            a, op = n.attr, n.op
            if op == "Identity": r = i[0]
            elif op == "FusedBatchNorm":
                xx, sc, b, m, var = i; r = (xx - m) / np.sqrt(var + a["epsilon"].f) * sc + b
            elif op == "Add": r = i[0] + i[1]
            elif op == "Shape": r = np.array(i[0].shape, np.int32)
            elif op == "Reshape": r = i[0].reshape([int(s) for s in i[1]])
            elif op == "Square": r = i[0] ** 2
            elif op == "Sum": r = i[0].sum(axis=tuple(np.atleast_1d(i[1]).tolist()), keepdims=bool(a["keep_dims"].b))
            elif op == "Sqrt": r = np.sqrt(i[0])
            elif op == "RealDiv": r = i[0] / i[1]
            else: raise NotImplementedError(op)
            v[n.name] = r
        return v["features"]
