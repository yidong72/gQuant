from ..util.contextCompositeNode import ContextCompositeNode
from gquant.dataframe_flow.portsSpecSchema import ConfSchema

__all__ = ["GridRandomSearchNode"]


class GridRandomSearchNode(ContextCompositeNode):

    def conf_schema(self):
        out_schema = ConfSchema(json={}, ui={})
        return out_schema
