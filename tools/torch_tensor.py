import torch
import torch.tensor


class TensorLoader():
    def __init__( self, np_seq ):
        self.np_seq = np_seq 
    def __iter__( self ):
        return self
    #def __next__(self):
    #    tup = self.np_seq.next()
    #    return tuple(map(torch.from_numpy, tup) )
    def __next__( self ):
        tup = self.np_seq.next()
        retx = torch.from_numpy( tup[0] )
        if len(tup) <= 1:
            return (retx)
        # Torch needs LongTensor
        rety = torch.from_numpy( tup[1].astype(int))
        # print(rety)
        return (retx, rety)
    def len(self):
        return self.np_seq.len()
    def __len__(self):
        return len(self.np_seq)
    
def to_loader( np_seq ):
    return TensorLoader( np_seq )