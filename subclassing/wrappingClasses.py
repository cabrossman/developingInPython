#https://rhettinger.wordpress.com/category/inheritance/

import logging
import collections
class LoggingDict(dict):
    def __setitem__(self, key, value):
        logging.info('Setting %s to %s' % (key, value))
        super().__setitem__(key, value)

lg = LoggingDict({'name' : 'chris', 'age' : 33})
lg['age'] = 34 #makes an entry in loggging.log file

class LoggingOD(LoggingDict, collections.OrderedDict):
    pass
    #The ancestor tree for our new class is: LoggingOD, LoggingDict, OrderedDict, dict, object.

######################################################################################################################
"""

For reorderable method calls to work, the classes need to be designed cooperatively. This presents three easily solved practical issues:

the method being called by super() needs to exist
the caller and callee need to have a matching argument signature -- use **kwds
and every occurrence of the method needs to use super()



"""

class Root: #highest level has functions
    def draw(self):
        # the delegation chain stops here
        assert not hasattr(super(), 'draw')

class Shape(Root):
    def __init__(self, shapename, **kwds):
        self.shapename = shapename
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting shape to:', self.shapename)
        super().draw()

class ColoredShape(Shape):
    def __init__(self, color, **kwds):
        self.color = color
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting color to:', self.color)
        super().draw()

cs = ColoredShape(color='blue', shapename='square')
cs.draw()


######################################################################################################################
"""
What about non cooperative classes
"""
#For example, the following Moveable class does not make super() calls, and it has an __init__() signature that is incompatible with object.__init__, and it does not inherit from Root:

class Moveable:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def draw(self):
        print('Drawing at position:', self.x, self.y)
#If we want to use this class with our cooperatively designed ColoredShape hierarchy, we need to make an adapter with the requisite super() calls:

class MoveableAdapter(Root):
    def __init__(self, x, y, **kwds):
        self.movable = Moveable(x, y)
        super().__init__(**kwds)
    def draw(self):
        self.movable.draw()
        super().draw()

class MovableColoredShape(ColoredShape, MoveableAdapter):
	pass

mcs = MovableColoredShape(color='red', shapename='triangle',x=10, y=20)
mcs.draw()


