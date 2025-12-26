import math,wave
import hashlib
import sys
import argparse
from fractions import Fraction
from array import array

T="""Triplet Letter Value Embedded
0 agm A None None
1 bnt B 4 None
2 co(UV(S|T)) C 2 UV(S|T)
3 dpv D 2/4 None
4 eqw E 3/4 None
5 fr(XY(Y|W)) F 3 XY(Y|W)
6 gsy G 3/8 None
7 hua H 64 None
8 iv(AB(A|Z)) I 9/4 AB(A|Z)
9 jwc J 12/4 None
10 kxd K 6/8 None
11 lye L 9/8 None
12 mzf M 12/8 None
"""

W={p[2]:(p[1],None if p[3]=="None" else Fraction(p[3]),"(" in l)
 for l in T.splitlines()[1:] if (p:=l.split())}

class Wheel:
    def __init__(self, table):
        self.table = table
        self.keys = sorted(table.keys())

    def program(self, x):
        code = ["def generated():", "    state = []"]
        for c in x.upper():
            entry = self.table.get(c)
            if not entry:
                continue
            triplet, val, embedded = entry
            logic = f"    state.append('{c}')  # triplet: {triplet}, value: {val}"
            if val is not None:
                logic += f"\n    print('Energy +{val}')"
            if embedded:
                logic += f"\n    print('Embedded gate logic active for {c}')"
            code.append(logic)
        code.append("    return state")
        return "\n".join(code)

    def fold(self, symbols):
        """Real folding operator - no code generation, just state transition"""
        next_state = []
        energy = Fraction(0, 1)

        for c in symbols:
            if c not in self.table:
                continue
                
            triplet, val, _ = self.table[c]  # embedded not used in folding
            
            # Rotate triplet core
            rotated_char = triplet[1] if len(triplet) > 1 else triplet[0]
            
            # Project back into symbol space
            projected = self.keys[ord(rotated_char) % len(self.keys)]
            next_state.append(projected)
            
            # Accumulate energy
            if val is not None:
                energy += val

        return next_state, energy

    def ingest_file(self, filepath):
        """Ingest file and extract symbols from SHA256 hash"""
        with open(filepath, 'rb') as f:
            data = f.read()
        
        digest = hashlib.sha256(data).hexdigest().upper()
        # Only keep symbols that exist in our wheel (A-M)
        valid_symbols = set(self.table.keys())
        symbols = [c for c in digest if c in valid_symbols]
        
        return symbols, len(data)

def main():
    parser = argparse.ArgumentParser(description='Generate code from file using Wheel')
    parser.add_argument('filepath', help='Path to file to process')
    args = parser.parse_args()
    
    try:
        wheel = Wheel(W)
        symbols, file_size = wheel.ingest_file(args.filepath)
        
        print(f"Processing file: {args.filepath}")
        print(f"File size: {file_size} bytes")
        print(f"Extracted symbols: {''.join(symbols)}")
        print(f"Symbol count: {len(symbols)}")
        print("\n--- Starting Real Folding Dynamics ---")
        print(f"Initial symbols: {''.join(symbols)}")
        
        # Real continuous folding
        state = symbols
        step = 0
        
        while True:
            state, energy = wheel.fold(state)
            print(f"energy={energy}, {''.join(state)}")
            step += 1
            
            # Break if state becomes empty (shouldn't happen with closure)
            if not state:
                print("State collapsed to empty")
                break
        
    except FileNotFoundError:
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
