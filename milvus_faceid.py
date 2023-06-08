import yaml
from milvus import (
    db,
    default_server,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_file_config(file_path):
    with open(file_path, "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    return opt


opt = load_file_config("./cfg.yaml")


# Create a Milvus connection
def create_connection(host, port, db_name):
    connections.connect(host=host, port=port)
    database = db.create_database(db_name)
    print(connections.list_connections())


def create_collection(dimensions, collection_name):
    code = FieldSchema(
        name="ID",
        dtype=DataType.VARCHAR,
        max_length=100,
        is_primary=True,
    )
    name = FieldSchema(
        name="Name_ID",
        dtype=DataType.VARCHAR,
        max_length=200,
    )
    department = FieldSchema(
        name="Department",
        dtype=DataType.VARCHAR,
        max_length=200,
    )
    vn_name = FieldSchema(
        name="Name_VN",
        dtype=DataType.VARCHAR,
        max_length=200,
    )
    embedder = FieldSchema(
        name="Embeddings", dtype=DataType.FLOAT_VECTOR, dim=dimensions
    )
    schema = CollectionSchema(
        fields=[code, name, department, vn_name, embedder],
        description="Faceid",
        enable_dynamic_field=True,
    )
    collection = Collection(
        name=collection_name, schema=schema, using="default", shards_num=2
    )
    return collection


# Drop a collection in Milvus
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))



