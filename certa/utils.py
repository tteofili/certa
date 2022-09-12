import pandas as pd


def get_row(r1, r2, lprefix='ltable_', rprefix='rtable_'):
    r1_df = pd.DataFrame(data=[r1.values], columns=r1.index)
    r2_df = pd.DataFrame(data=[r2.values], columns=r2.index)
    r1_df.columns = list(map(lambda col: lprefix + col, r1_df.columns))
    r2_df.columns = list(map(lambda col: rprefix + col, r2_df.columns))
    r1r2 = pd.concat([r1_df, r2_df], axis=1)
    return r1r2


def merge_sources(table, left_prefix, right_prefix, left_source, right_source, copy_from_table, ignore_from_table,
                  robust: bool = False):
    dataset = pd.DataFrame(columns={col: table[col].dtype for col in copy_from_table})
    ignore_column = copy_from_table + ignore_from_table

    for _, row in table.iterrows():
        leftid = row[left_prefix + 'id']
        rightid = row[right_prefix + 'id']
        l_tuple = left_source.loc[left_source['id'] == leftid].iloc[0]
        r_tuple = right_source.loc[right_source['id'] == rightid].iloc[0]
        for ic in ignore_column:
            if ic in l_tuple:
                l_tuple = l_tuple.drop([ic])
            if ic in r_tuple:
                r_tuple = r_tuple.drop([ic])
        new_row = get_row(l_tuple, r_tuple, lprefix=left_prefix, rprefix=right_prefix)
        new_row['label'] = row['label']
        dataset = dataset.append(new_row, ignore_index=True)

    if robust:
            # symmetry
            sym_new_row = {column: row[column] for column in copy_from_table}
            try:
                for id, source, prefix in [(rightid, right_source, left_prefix), (leftid, left_source, right_prefix)]:

                    for column in source.keys():
                        if column not in ignore_column:
                            sym_new_row[prefix + column] = source.loc[id][column]

                dataset = dataset.append(sym_new_row, ignore_index=True)
            except:
                pass

            # identity
            lcopy_row = {column: row[column] for column in copy_from_table}
            try:
                for id, source, prefix in [(leftid, left_source, left_prefix), (leftid, left_source, right_prefix)]:

                    for column in source.keys():
                        if column not in ignore_column:
                            lcopy_row[prefix + column] = source.loc[id][column]

                lcopy_row['label'] = 1
                dataset = dataset.append(lcopy_row, ignore_index=True)
            except:
                pass

            rcopy_row = {column: row[column] for column in copy_from_table}
            try:
                for id, source, prefix in [(rightid, right_source, left_prefix), (rightid, right_source, right_prefix)]:

                    for column in source.keys():
                        if column not in ignore_column:
                            rcopy_row[prefix + column] = source.loc[id][column]

                rcopy_row['label'] = 1
                dataset = dataset.append(rcopy_row, ignore_index=True)
            except:
                pass
    return dataset


def diff(a: str, b: str):
    d = set(a.split(' ')).difference(b.split(' '))
    if len(d) == 0:
        d = '+' + str(set(b.split(' ')).difference(a.split(' ')))
    else:
        d = '-' + str(d)
    return d


class lattice(object):

    def __init__(self, Uelements, ranks, join_func=lambda a,b : a.union(b), meet_func=lambda a,b : a.intersection(b),
                 triangle=pd.DataFrame()):
        '''Create a lattice:

        Keyword arguments:
        Uelements -- list. The lattice set.
        ranks -- list. The lattice ranking.
        join_func  -- join function that operates to elements and returns the greatest element.
        meet_func  -- meet function that operates to elements and returns the least element.

        Returns a lattice instance.
        '''
        self.Uelements = Uelements
        self.ranks = ranks
        self.join=join_func
        self.meet=meet_func
        self.triangle = triangle

    def wrap(self,object):
        '''Wraps an object as a lattice element:

        Keyword argument:
        object -- any item from the lattice set.
        '''
        return LatticeElement(self,object)

    def WElementByIndex(self,ElementIndex):
        return LatticeElement(self,self.Uelements[ElementIndex])

    @property
    def TopElement(self):
        top=self.wrap(self.Uelements[0])
        for element in self.Uelements[1:]:
            top |= self.wrap(element)
        return top

    @property
    def BottonElement(self):
        botton=self.wrap(self.Uelements[0])
        for element in self.Uelements[1:]:
            botton &= self.wrap(element)
        return botton

    def hasse(self, depth=-1, compress=False):
        graph=dict()
        matching = []
        non_matching = []
        for indexS,elementS in enumerate(self.Uelements):
            graph[indexS]=[]
            for indexD,elementD in enumerate(self.Uelements):
                if self.wrap(elementS) <= self.wrap(elementD):
                    if not bool( sum([ int(self.WElementByIndex(x) <= self.wrap(elementD)) for x in graph[indexS]])) and not elementS==elementD:
                        graph[indexS]+=[indexD]
        dotcode='digraph G {\nsplines="line"\nrankdir=BT\n'
        topebi = str(self.TopElement.unwrap)
        if compress:
            topebi = compress_text(topebi)
        dotcode+='\"' + topebi + '\" [shape=box];\n'
        bebi = str(self.BottonElement.unwrap)
        if compress:
            bebi = compress_text(bebi)
        dotcode+='\"' + bebi + '\" [shape=box];\n'
        dc = 0
        for s, ds in graph.items():
            ebi = str(self.WElementByIndex(s).unwrap)
            if compress:
                ebi = compress_text(ebi)
            color = ''
            if not ebi in matching:
                if self.ranks[s] > 0.5:
                    color = 'green'
            if not ebi in non_matching:
                if self.ranks[s] < 0.5:
                    color = 'red'
            dotcode += "\""+ebi+"\" [color="+color+"];\n"
            for d in ds:
                dsebi = str(self.WElementByIndex(d))
                if compress:
                    dsebi = compress_text(dsebi)
                dotcode += "\""+ebi+"\""
                dotcode += " -> "
                dotcode += "\"" + dsebi + "\""
                dotcode += ";\n"
            dc+=1
            if depth > 0 and dc==depth:
                break
        dotcode += "}"
        return dotcode

    def __repr__(self):
        """Represents the lattice as an instance of Lattice."""
        return 'Lattice(%s,%s,%s)' % (self.Uelements,self.join,self.meet)


def compress_text(s):
    attrs = []
    for attr in s.replace('{', '').replace('}', '').split(','):
        parts = []
        for part in attr.split('_'):
            if 'ltable' in part:
                parts.append('L')
            elif 'rtable' in part:
                parts.append('R')
            else:
                parts.append(part[:3].replace('a','').replace('e','').replace('i','').replace('o','').replace('u',''))
        attrs.append(parts)
    return '_'.join(parts)


class LatticeElement():
    def __init__(self, lattice, Uelement):
        if Uelement not in lattice.Uelements: lattice.Uelements.append(Uelement)
        self.lattice=lattice
        self.ElementIndex=lattice.Uelements.index(Uelement)

    @property
    def unwrap(self):
        return self.lattice.Uelements[self.ElementIndex]

    def __str__(self):
        return str(self.unwrap)

    def __repr__(self):
        """Represents the lattice element as an instance of LatticeElement."""
        return "LatticeElement(L, %s)" % str(self)

    def __and__(self,b):
        # a.__and__(b) <=> a & b <=> meet(a,b)
        return LatticeElement(self.lattice,self.lattice.meet(self.unwrap,b.unwrap))

    def __or__(self,b):
        # a.__or__(b) <=> a | b <=> join(a,b)
        return LatticeElement(self.lattice,self.lattice.join(self.unwrap,b.unwrap))

    def __eq__(self,b):
        # a.__eq__(b) <=> a = b <=> join(a,b)
        return self.unwrap==b.unwrap

    def __le__(self,b):
        # a <= b if and only if a = a & b,
        # or
        # a <= b if and only if b = a | b,
        a=self
        return ( a == a & b ) or ( b == a | b )