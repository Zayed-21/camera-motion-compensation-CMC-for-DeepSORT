"""Make DeepSORT's released appearance model (mars-small128.pb, Wojke et al.) loadable by
OpenCV DNN. The graph's only preprocessing is uint8->float Cast followed by a tf.map_fn
that reverses channel order (BGR->RGB, slice [::1, ::1, ::-1], verified from the graph's
constants). OpenCV cannot run TensorArray loops, so we drop those nodes and feed a float
RGB placeholder straight into conv1_1. Weights are untouched (unknown protobuf fields
round-trip byte for byte). Callers must therefore pass RGB (swapRB=True), 0-255 floats."""
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

f = descriptor_pb2.FileDescriptorProto(name="mini_graph.proto", package="mini", syntax="proto3")
av = f.message_type.add(name="AttrValue")
av.field.add(name="type", number=6, type=5, label=1)
dim = f.message_type.add(name="Dim"); dim.field.add(name="size", number=1, type=3, label=1)
shp = f.message_type.add(name="Shape"); shp.field.add(name="dim", number=2, type=11, label=3, type_name=".mini.Dim")
av.field.add(name="shape", number=7, type=11, label=1, type_name=".mini.Shape")
nd = f.message_type.add(name="NodeDef")
nd.field.add(name="name", number=1, type=9, label=1)
nd.field.add(name="op", number=2, type=9, label=1)
nd.field.add(name="input", number=3, type=9, label=3)
ent = nd.nested_type.add(name="AttrEntry"); ent.options.map_entry = True
ent.field.add(name="key", number=1, type=9, label=1)
ent.field.add(name="value", number=2, type=11, label=1, type_name=".mini.AttrValue")
nd.field.add(name="attr", number=5, type=11, label=3, type_name=".mini.NodeDef.AttrEntry")
gd = f.message_type.add(name="GraphDef")
gd.field.add(name="node", number=1, type=11, label=3, type_name=".mini.NodeDef")
pool = descriptor_pool.DescriptorPool(); pool.Add(f)
Graph = message_factory.GetMessageClass(pool.FindMessageTypeByName("mini.GraphDef"))
GATHER = "map/TensorArrayStack/TensorArrayGatherV3"

def fix(src, dst):
    g = Graph(); g.ParseFromString(open(src, "rb").read())
    keep = [n for n in g.node if not (n.name in ("images", "Cast") or n.name.startswith("map/"))]
    users = [n for n in keep if GATHER in n.input]
    assert [n.name for n in users] == ["conv1_1/Conv2D"], users
    out = Graph()
    ph = out.node.add(name="input_rgb", op="Placeholder"); ph.attr["dtype"].type = 1
    for d in (-1, 128, 64, 3):
        ph.attr["shape"].shape.dim.add(size=d)
    for n in keep:
        m = out.node.add(); m.CopyFrom(n)
        if n.name == "conv1_1/Conv2D":
            m.input[0] = "input_rgb"
    open(dst, "wb").write(out.SerializeToString())
    return len(g.node), len(out.node)

if __name__ == "__main__":
    print(fix("models/mars-small128.pb", "models/mars-small128-cv.pb"))

def truncate(src, dst, last="fc1/Elu"):
    """Keep nodes up to `last` (graph is topologically ordered); OpenCV cannot import
    the final 'ball' reshape. The ball batch-norm + L2 step is done in numpy."""
    g = Graph(); g.ParseFromString(open(src, "rb").read())
    out = Graph()
    for n in g.node:
        out.node.add().CopyFrom(n)
        if n.name == last: break
    open(dst, "wb").write(out.SerializeToString())
