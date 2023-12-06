class Hierachy:

    """
    Definition: This is the hierachy object, which will represent the
    child/parent relationship between the various objects with the passed
    in mask RCNN object

    Parameters:
    1. polygon: this is the polygon that
    will represent the current object in the hierachy
    2. objectLabel: this is the label for the current object, such as bus

    Returns: Nothing
    """
    def __init__(self, polygon, object_label, description):
        """
        Fields
        1. polygon: this is the polygon that we will be saving
        2. children: this is the children objects of the current object of the hierachy
        3. object_label: this is the label for the current object, such as bus
        4. text: this is the text that the current object has
        5. descriptions: these are the descriptions that the current object has
        6. is_child: tells us if the current object is a child of another object
        7. includes: this is used to tell us if we will be including a hierachy in the final answer or not
        """
        self.polygon = polygon
        self.children = None
        self.object_label = object_label
        self.text = {}
        self.descriptions = {"description" : description}
        self.is_child = False
        self.include = True

    """
    Definition: This gives the size of the current hierachy 

    Parameters: nothing

    Returns: the size of the hierachy
    """
    @staticmethod
    def hierachy_size(hierachy):
        if hierachy == None:
            return 0

        if hierachy.children == None:
            return 1

        maxSize = 0
        for node in hierachy.children:
            nodeSize = hierachy.hierachy_size(node)
            maxSize = max((1 + nodeSize), maxSize)

        return maxSize
