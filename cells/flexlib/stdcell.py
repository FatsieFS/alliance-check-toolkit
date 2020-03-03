import itertools
from six import add_metaclass
from abc import ABCMeta, abstractmethod

from Hurricane import (
    NetExternalComponents, UpdateSession,
    DataBase, Cell, Net, Contact, Horizontal, Vertical, Box
)

__all__ = ["BBox", "Wire", "Via", "Device", "StdCell"]

class BBox(object):
    def __init__(self, x1, y1, x2, y2):
        assert x1 <= x2 and y1 <= y2
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __repr__(self):
        return "({},{})-({},{})".format(self.x1, self.y1, self.x2, self.y2)

    def overlaps(self, box):
        return ((self.x2 >= box.x1)
                and (box.x2 >= self.x1)
                and (self.y2 >= box.y1)
                and (box.y2 >= self.y1)
               )

    def encloses(self, box):
        return ((self.x1 <= box.x1)
                and (self.y1 <= box.y1)
                and (self.x2 >= box.x2)
                and (self.y2 >= box.y2)
               )

    def copy(self):
        return self.__class__(self.x1, self.y1, self.x2, self.y2)

class _LayerBox(BBox):
    def __init__(self, layer, x1, y1, x2, y2):
        super(_LayerBox, self).__init__(x1, y1, x2, y2)
        self.layer = layer
        
    def __repr__(self):
        return "{}({})".format(self.layer, super(_LayerBox, self).__repr__())
        
    def overlaps(self, box):
        return (self.layer == box.layer) and super(_LayerBox, self).overlaps(box)
    
    def encloses(self, box):
        return (self.layer == box.layer) and super(_LayerBox, self).encloses(box)

    def copy(self):
        return self.__class__(self.layer, self.x1, self.y1, self.x2, self.y2)

class _UniqueNetName(object):
    def __init__(self):
        self.netnr = 0

    def new(self):
        s = "*{:04d}".format(self.netnr)
        self.netnr += 1
        return s

class _LayerBoxesNets(object):
    def __init__(self):
        self._layerboxes = {}
        self._netaliases = {}
        self._uniquenet = _UniqueNetName()

    def add_alias(self, net1, net2):
        assert (net1 != "fused_net") and (net1 in self._netaliases)
        net1 = self.from_alias(net1)
        if net2 == "fused_net":
            net2 = self._uniquenet.new()
        if net2 not in self._netaliases:
            if net2[0] == "*":
                return net1
            else:
                assert net1[0] == "*", "Shorted net {} and {}".format(net1, net2)
                self._netaliases[net1] = net2
                # It can be that net2 is not in aliases when called for first time
                if net2 not in self._netaliases:
                    self._netaliases[net2] = net2
                return net2
        else:
            # net1 and net2 are there, really join them
            net2 = self.from_alias(net2)
            if net1 == net2:
                net = net1
            elif net2[0] != "*":
                assert net1[0] == "*", "Shorted nets {} and {}".format(net1, net2)
                net = net2
                self._netaliases[net1] = net
            else:
                net = net1
                self._netaliases[net2] = net

            return net

    def from_alias(self, net):
        while self._netaliases[net] != net:
            net = self._netaliases[net]
        return net

    def finalize_nets(self):
        starnets = set()
        for net in self._netaliases.keys():
            net2 = self.from_alias(net)
            if net2[0] == "*":
                starnets.add(net2)
        newnets = dict(
            (net, "_net{}".format(i)) for i, net in enumerate(starnets)
        )
        self._netaliases.update(newnets)
        self._netaliases.update(dict(
            (net, net) for net in newnets.values()
        ))
        return set(self.from_alias(net) for net in self._netaliases)

    def add_box(self, net, box):
        layer = box.layer
        try:
            boxes = self._layerboxes[layer]
        except KeyError:
            self._layerboxes[layer] = boxes = []

        for box2, net2 in boxes:
            if box.overlaps(box2):
                net = self.add_alias(net2, net)

        if net == "fused_net":
            # Get name for unnamed net
            net = self._uniquenet.new()
        if net not in self._netaliases:
            self._netaliases[net] = net

        boxes.append((box, net))

        return net

@add_metaclass(ABCMeta)
class _Element(object):
    # Sizing parameters for bounding box derivation

    def __init__(self, external, boxes):
        self.external = external
        self._ignore = False
        self.boxes = boxes
       
        self.connects = {}
    
    def _str_indent(self, level, str_elem, prefix, level_str, net):
        s = level*level_str + prefix + str_elem
        
        if net is None:
            if len(self.connects) > 0:
                for subnet, elems in self.connects.items():
                    s += "\n{}{}Net: {}".format((level + 1)*level_str, prefix, subnet)
                    for elem in elems:
                        s += "\n"+elem.str_indent(level+2, prefix=prefix, level_str=level_str, net=subnet)
        elif net in self.connects:
            for elem in self.connects[net]:
                s += "\n"+elem.str_indent(level+1, prefix=prefix, level_str=level_str, net=net)
    
        return s

    @abstractmethod
    def str_indent(self, level, prefix="", level_str="  ", net=None):
        raise NotImplementedError("Abstract method not implemented")
    
    def __str__(self):
        return self.str_indent(0)
    
    def overlaps(self, other):
        for box1, box2 in itertools.product(self.boxes, other.boxes):
            if box1.overlaps(box2):
                return True
        
        return False
    
    def add_boxes(self, layerboxes):
        if hasattr(self, "net"):
            for box in self.boxes:
                self.net = layerboxes.add_box(self.net, box)
        else:
            for i, box in enumerate(self.boxes):
                self.nets[i] = layerboxes.add_box(self.nets[i], box)

    def add_connects(self, net, connects):
        self.connects[net] = connects

    def iterate_connects(self, net=None, include_ignored=False):
        stack = [self.connects]
        
        while stack:
            connects = stack.pop()
            for elems_net, elems in connects.items():
                if (net is None) or (net == elems_net):
                    for elem in elems:
                        if (not elem._ignore) or include_ignored:
                            yield elem
                        stack.append(elem.connects)

    def get_nets(self):
        try:
            nets = [self.net]
        except AttributeError:
            nets = self.nets
        
        return nets

    def update_nets(self, layerboxes):
        try:
            self.net = layerboxes.from_alias(self.net)
        except AttributeError:
            self.nets = [layerboxes.from_alias(net) for net in self.nets]

    def _merge(self, elem):
        return False
    
    def merge(self, elem):
        if self._ignore or elem._ignore:
            return False
        else:
            return self._merge(elem) or elem._merge(self)

    def create_in_coriolis(self, net):
        netname = net.getName()
        
        if netname in self.connects:
            for elem in self.connects[netname]:
                elem.create_in_coriolis(net)

    @abstractmethod
    def python_code(self):
        raise NotImplementedError("Abstract method not implemented")

class Wire(_Element):
    layers = ("NWELL", "NTIE", "PTIE", "NDIF", "PDIF", "POLY", "METAL1", "METAL2", "METAL3")
    dhw = {
        "NWELL": 0, "NTIE": 200, "PTIE": 200, "NDIF": 200, "PDIF": 200,
        "POLY": 0, "METAL1": 0, "METAL2": 0, "METAL3": 0
    }
    dhl = {
        "NWELL": 0, "NTIE": 200, "PTIE": 200, "NDIF": 200, "PDIF": 200,
        "POLY": 200, "METAL1": 200, "METAL2": 200, "METAL3": 200
    }

    def __init__(self, layer, x, y, width, external=False):
        assert isinstance(x, tuple) ^ isinstance(y, tuple)
        self.layer = layer
        self.x = x
        self.y = y
        self.width = width

        dhw = self.dhw[layer]
        dhl = self.dhl[layer]
        if not isinstance(x, tuple):
            x1 = x - width//2 - dhw
            x2 = x + width//2 + dhw
        else:
            x1 = x[0] - dhl
            x2 = x[1] + dhl
        if not isinstance(y, tuple):
            y1 = y - width//2 - dhw
            y2 = y + width//2 + dhw
        else:
            y1 = y[0] - dhl
            y2 = y[1] + dhl
        boxes = [_LayerBox(layer, x1, y1, x2, y2)]
        if layer == "NTIE": # Make NWELL connection
            boxes.append(_LayerBox("NWELL", x1, y1, x2, y2))
        super(Wire, self).__init__(external, boxes)

    def _merge(self, elem):
        if not (isinstance(elem, Wire) and (self.layer == elem.layer)):
            return False

        merged = False

        box_self = self.boxes[0]
        box_elem = elem.boxes[0]
        hor_self = isinstance(self.x, tuple)
        hor_elem = isinstance(elem.x, tuple)
        if box_self.encloses(box_elem):
            self.external |= elem.external
            elem._ignore = True
            merged = True
        elif box_elem.encloses(box_self):
            self._ignore = True
            elem.external |= self.external
            merged = True
        elif (not hor_self) and (not hor_elem): # Both vertical
            if (self.x == elem.x) and (self.width == elem.width) and (self.y[1] >= elem.y[0]) and (elem.y[1] >= self.y[0]):
                self.y1 = box_self.y1 = min(box_self.y1, box_elem.y1)
                self.y2 = box_self.y2 = max(box_self.y2, box_elem.y2)
                self.y = (min(self.y[0], elem.y[0]), max(self.y[1], elem.y[1]))
                self.external |= elem.external
                elem._ignore = True
                merged = True
        elif hor_self and hor_elem: # Both horizontal
            if (self.y == elem.y) and (self.width == elem.width) and (self.x[1] >= elem.x[0]) and (elem.x[1] >= self.x[0]):
                self.x1 = box_self.x1 = min(box_self.x1, box_elem.x1)
                self.x2 = box_self.x2 = max(box_self.x2, box_elem.x2)
                self.x = (min(self.x[0], elem.x[0]), max(self.x[1], elem.x[1]))
                self.external |= elem.external
                elem._ignore = True
                merged = True

        if merged:
            # Some segments may have more than one box for connectivity, f.ex. NTIE with NWELL
            for box in self.boxes[1:]:
                box.x1 = box_self.x1
                box.x2 = box_self.x2
                box.y1 = box_self.y1
                box.y2 = box_self.y2
            for box in elem.boxes[1:]:
                box.x1 = box_elem.x1
                box.x2 = box_elem.x2
                box.y1 = box_elem.y1
                box.y2 = box_elem.y2

        return merged

    def str_indent(self, level, prefix="", level_str="  ", net=None):
        box = self.boxes[0]
        str_elem = "{}{}(({},{})-({},{}))".format(
            "EXT_" if self.external else "",
            self.layer,
            box.x1, box.y1, box.x2, box.y2,
        )

        return self._str_indent(level, str_elem, prefix, level_str, net)

    def create_in_coriolis(self, net):
        tech = DataBase.getDB().getTechnology()
        netname = net.getName()

        layer = tech.getLayer(self.layer)
        if isinstance(self.x, tuple):
            comp = Horizontal.create(net, layer, self.y, self.width, self.x[0], self.x[1])
        elif isinstance(self.y, tuple):
            comp = Vertical.create(net, layer, self.x, self.width, self.y[0], self.y[1])
        else:
            raise Exception("net {}: unhandled elem coordinate specification x={}, y={}".format(netname, self.x, self.y))
        if self.external:
            NetExternalComponents.setExternal(comp)

        super(Wire, self).create_in_coriolis(net)

    def python_code(self, lookup={}):
        classstr = lookup.get("Wire", "Wire")
        return "{}({!r}, {!r}, {!r}, {!r}, {!r})".format(classstr, self.layer, self.x, self.y, self.width, self.external)

class Via(_Element):
    layer2bottom = {
        "CONT_BODY_N": "NTIE",
        "CONT_BODY_P": "PTIE",
        "CONT_DIF_N": "NDIF",
        "CONT_DIF_P": "PDIF",
        "CONT_POLY": "POLY",
    }
    layer2top = {
        "CONT_BODY_N": "METAL1",
        "CONT_BODY_P": "METAL1",
        "CONT_DIF_N": "METAL1",
        "CONT_DIF_P": "METAL1",
        "CONT_POLY": "METAL1",
    }
    bottom2layer = {
        "NTIE": "CONT_BODY_N",
        "PTIE": "CONT_BODY_P",
        "NDIF": "CONT_DIF_N",
        "PDIF": "CONT_DIF_P",
        "POLY": "CONT_POLY",
    }
    dhw = {
        "NTIE": 600,
        "PTIE": 600,
        "NDIF": 600,
        "PDIF": 600,
        "POLY": 600,
        "METAL1": 400,
    }

    def __init__(self, bottom, top, x, y, width):
        self.x = x
        self.y = y
        self.width = width
        self.bottom = bottom
        self.top = top

        dhw_bottom = width//2 + self.dhw[bottom]
        dhw_top = width//2 + self.dhw[top]
        boxes = [
            _LayerBox(bottom, x - dhw_bottom, y - dhw_bottom, x + dhw_bottom, y + dhw_bottom),
            _LayerBox(top, x - dhw_top, y - dhw_top, x + dhw_top, y + dhw_top),
        ]

        super(Via, self).__init__(False, boxes)

    def str_indent(self, level, prefix="", level_str="  ", net=None):
        str_elem = "{}<->{}(({},{}))".format(
            self.bottom, self.top,
            self.x, self.y,
        )

        return self._str_indent(level, str_elem, prefix, level_str, net)

    def create_in_coriolis(self, net):
        tech = DataBase.getDB().getTechnology()

        assert self.top == "METAL1"
        layer = tech.getLayer(self.bottom2layer[self.bottom])
        Contact.create(net, layer, self.x, self.y, 200, 200)

        super(Via, self).create_in_coriolis(net)

    def python_code(self, lookup={}):
        classstr = lookup.get("Via", "Via")
        return "{}({!r}, {!r}, {!r}, {!r}, {!r})".format(classstr, self.bottom, self.top, self.x, self.y, self.width)

class Device(_Element):
    layer2type = {
        "NTRANS": "nmos",
        "PTRANS": "pmos",
    }
    type2layer = {
        "nmos": "NTRANS",
        "pmos": "PTRANS",
    }
    type2gatelayer = {
        "nmos": "POLY",
        "pmos": "POLY",
    }
    type2difflayer = {
        "nmos": "NDIF",
        "pmos": "PDIF",
    }
    dhl = 0
    dhw = 400
    diffwidth = 600

    def __init__(self, type_, x, y, l, w, direction, source_net="fused_net", drain_net="fused_net"):
        assert type_ in ("nmos", "pmos")
        self.type = type_
        self.x = x
        self.y = y
        self.l = l
        self.w = w
        assert direction in ("vertical") # Todo support horizontal transistors
        self.direction = direction
        self.source = source = {"net": source_net}
        self.drain = drain = {"net": drain_net}
        difflayer = self.type2difflayer[type_]
        dhl = l//2 + self.dhl
        dhw = w//2 + self.dhw
        x1_gate = x - dhl
        x2_gate = x + dhl
        y1_gate = y - dhw
        y2_gate = y + dhw
        x2_source = x1_gate
        x1_source = x2_source - self.diffwidth
        y1_source = y - w//2
        y2_source = y + w//2
        x1_drain = x2_gate
        x2_drain = x1_drain + self.diffwidth
        y1_drain = y1_source
        y2_drain = y2_source
        boxes = [_LayerBox(self.type2gatelayer[type_], x1_gate, y1_gate, x2_gate, y2_gate)]
        source["box"] = _LayerBox(difflayer, x1_source, y1_source, x2_source, y2_source)
        drain["box"] = _LayerBox(difflayer, x1_drain, y1_drain, x2_drain, y2_drain)
        super(Device, self).__init__(False, boxes)

    def str_indent(self, level, prefix="", level_str="  ", net=None):
        str_elem = "{}(({},{}),l={},w={})".format(
            self.type,
            self.x, self.y,
            self.l, self.w,
        )

        return self._str_indent(level, str_elem, prefix, level_str, net)

    def create_in_coriolis(self, net):
        tech = DataBase.getDB().getTechnology()

        layer = tech.getLayer(self.type2layer[self.type])
        width = self.l
        height = self.w
        x = self.x
        y = self.y
        Vertical.create(net, layer, x, width, y - height//2, y + height//2)

        super(Device, self).create_in_coriolis(net)

    def python_code(self, lookup={}):
        classstr = lookup.get("Device", "Device")
        return "{}({!r}, {!r}, {!r}, {!r}, {!r}, {!r}, source_net={!r}, drain_net={!r})".format(
            classstr, self.type, self.x, self.y, self.l, self.w, self.direction, self.source["net"], self.drain["net"],
        )

class StdCell(object):
    def __init__(self, name="NoName", width=0, height=0, nets={}, finalize=False):
        self.name = name
        self.width = width
        self.height = height
        self.ports = set()
        self.nets = {}

        self._layerboxesnets = _LayerBoxesNets()

        elem = Wire("METAL1", (0, width), 1200, 2400, external=True)
        elem._ignore = True
        self.add_elem(elem, net="vss")
        elem = Wire("METAL1", (0, width), 18800, 2400, external=True)
        elem._ignore = True
        self.add_elem(elem, net="vdd")
        elem = Wire("NWELL", (-600, width+600), 15600, 12000)
        elem._ignore = True
        self.add_elem(elem, net="vdd")

        for net, elems in nets.items():
            for elem in elems:
                self.add_elem(elem, net)

        if finalize:
            self.finalize()

    def add_elem(self, elem, net="fused_net"):
        assert self._layerboxesnets is not None, "add_elem() called on cell {} in finalized state".format(self.name)

        for box in elem.boxes:
            net = self._layerboxesnets.add_box(net, box)
        if isinstance(elem, Device):
            elem.source["net"] = self._layerboxesnets.add_box(elem.source["net"], elem.source["box"])
            elem.drain["net"] = self._layerboxesnets.add_box(elem.drain["net"], elem.drain["box"])
        
        try:
            self.nets[net].append(elem)
        except KeyError:
            self.nets[net] = [elem]
        if elem.external:
            self.ports |= {net}

    def _add_elem2net(self, net, elem):
        try:
            self.nets[net].append(elem)
        except KeyError:
            self.nets[net] = [elem]

    def _add_elems2net(self, net, elems):
        assert isinstance(elems, list)
        try:
            self.nets[net] += elems
        except KeyError:
            self.nets[net] = elems

    def _update_devicenets(self, elems):
        for elem in elems:
            if isinstance(elem, Device):
                source_net = elem.source["net"]
                elem.source["net"] = new_net = self._layerboxesnets.from_alias(source_net)
                try:
                    elem.connects[new_net] = elem.connects.pop(source_net)
                except KeyError:
                    pass

                drain_net = elem.drain["net"]
                elem.drain["net"] = new_net = self._layerboxesnets.from_alias(drain_net)
                try:
                    elem.connects[new_net] = elem.connects.pop(drain_net)
                except KeyError:
                    pass

            for _, elems in elem.connects.items():
                self._update_devicenets(elems)

    @staticmethod
    def _connect_elem(net, elem, todo):
        # Try to connect elem in todo set of elems and remove the connected ones from todo
        conns = set(filter(lambda other: elem.overlaps(other), todo))
        if conns:
            todo -= conns
            map(lambda elem: StdCell._connect_elem(net, elem, todo), conns)
            elem.add_connects(net, list(conns))

    def iterate_net(self, net, include_ignored=False):
        for elem in self.nets[net]:
            if (not elem._ignore) or include_ignored:
                yield elem
            for elem2 in elem.iterate_connects(net, include_ignored=include_ignored):
                yield elem2

    def finalize(self):
        netnames = self._layerboxesnets.finalize_nets()
        retval = {}

        # Add elems in nets that disappeared to the final net
        removednets = set(self.nets.keys()) - netnames
        for net in removednets:
            self._add_elems2net(self._layerboxesnets.from_alias(net), self.nets.pop(net))
        # Check that all ports have a net associated with it
        assert self.ports.issubset(set(self.nets.keys()))

        # Update the net names
        for elems in self.nets.values():
            self._update_devicenets(elems)

        # Merge elems in a net if possible
        merged = 0
        for net in self.nets.keys():
            merged += len(filter(
                lambda (elem1, elem2): elem1.merge(elem2),
                itertools.combinations(self.iterate_net(net), 2)
            ))
        retval["merged"] = merged

        # (Re)connect the overlapping interconnects in a net
        # Do ignore the ignored elems
        for net, elems in self.nets.items(): # Only connect within the same net.
            tops = []
            # Set todo to all elems in the net that are not ignored
            todo = set(self.iterate_net(net))
            assert len(todo) > 0 or net in ("vss", "vdd"), "empty todo for net {}".format(net)
            while len(todo) > 0:
                elem = None
                # First search for external net for a port
                for it in todo:
                    if it.external:
                        elem = it
                        break
                # Then for a METAL1 segment
                if elem is None:
                    for it in todo:
                        if isinstance(it, Wire) and (it.layer == "METAL1"):
                            elem = it
                            break
                # Then first non-device
                if elem is None:
                    for it in todo:
                        if not isinstance(it, Device):
                            elem = it
                            break
                if elem is None:
                    elem = todo.pop()
                else:
                    # Remove selected elem
                    todo -= {elem}
                self._connect_elem(net, elem, todo)
                tops.append(elem)
            # Only retain the top elements in elems
            elems[:] = tops[:]
            #assert len(elems) == 1 or net in ("vss", "vdd")
            if len(elems) != 1 and net not in ("vss", "vdd"):
                print("{} has {} top elems on net {}".format(self.name, len(elems), net))

        self._layerboxesnets = None

        return retval

    def create_in_coriolis(self, library=None):
        tech = DataBase.getDB().getTechnology()

        if library is None:
            import CRL

            # Use Alliance framework
            framework = CRL.AllianceFramework.get()
            cell = framework.createCell(self.name)
        else:
            cell = Cell.create(library, self.name)

        UpdateSession.open()

        # Create the nets
        nets = {}
        for name in self.nets.keys():
            nets[name] = net = Net.create(cell, name)
            net.setExternal(name in self.ports)
        # Template nets may be left out of the net list
        for name in ("vss", "vdd"):
            if name not in nets:
                net = Net.create(cell, name)
                net.setExternal(True)
                nets[name] = net

        # Template segments
        cellwidth = self.width
        cell.setAbutmentBox(Box(0, 0, cellwidth, self.height))
        comp = Horizontal.create(nets["vss"], tech.getLayer("METAL1"), 1200, 2400, 0, cellwidth)
        NetExternalComponents.setExternal(comp)
        comp = Horizontal.create(nets["vdd"], tech.getLayer("METAL1"), 18800, 2400, 0, cellwidth)
        NetExternalComponents.setExternal(comp)
        comp = Horizontal.create(nets["vdd"], tech.getLayer("NWELL"), 15600, 12000, -600, cellwidth+600)

        # All other segments
        for netname, elems in self.nets.items():
            for elem in elems:
                elem.create_in_coriolis(nets[netname])

        UpdateSession.close()

        if library is None:
            framework.saveCell(cell, CRL.Catalog.State.Physical)

    def python_code(self, level=0, level_str="    ", lookup={}):
        classstr = lookup.get("StdCell", "StdCell")

        def indent_str():
            return level*level_str
        
        s = indent_str() + classstr + "(\n"
        level += 1
        s += indent_str() + "name={!r}, width={!r}, height={!r},\n".format(
            self.name, self.width, self.height,
        )
        s += indent_str() + "nets={\n"
        netnames = self.nets.keys()
        netnames.sort()
        level += 1
        for net in netnames:
            s += indent_str() + "{!r}: [\n".format(net)
            level += 1
            for elem in self.iterate_net(net):
                s += indent_str() + "{},\n".format(elem.python_code(), lookup=lookup)
            level -= 1
            s += indent_str() + "],\n"
        level -= 1
        s += indent_str() + "},\n"
        if not self._layerboxesnets:
            s += indent_str() + "finalize=True,\n"
        level -= 1
        s += indent_str() + ")"

        return s