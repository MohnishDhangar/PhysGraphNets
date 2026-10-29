"""Canonical physical identities exclude mesh resolution, labels and split names."""
from dataclasses import dataclass
import hashlib
import json
import math


def canonical_value(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int,float)):
        if not math.isfinite(value):
            raise ValueError('instance parameters must be finite')
        return 0. if value == 0 else float(value)
    if isinstance(value, (list,tuple)):
        return [canonical_value(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k,str) and k for k in value):
        return {k:canonical_value(v) for k,v in value.items()}
    raise ValueError('instance parameters must be JSON-compatible values with string keys')


@dataclass(frozen=True)
class PDEInstance:
    canonical_json: str

    @classmethod
    def create(cls, family, geometry, physics):
        if not isinstance(family,str) or not family.strip() or not isinstance(geometry,dict) or not isinstance(physics,dict):
            raise ValueError('require a named family and geometry/physics dictionaries')
        payload = {'schema_version':1,'family':family,'geometry':canonical_value(geometry),'physics':canonical_value(physics)}
        return cls(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False))

    @property
    def instance_id(self):
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


def split_instances(instances, *, seed=0, train_fraction=.7, validation_fraction=.15):
    if isinstance(seed,bool) or not isinstance(seed,int) or seed < 0:
        raise ValueError('split seed must be a nonnegative integer')
    if (not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                for v in (train_fraction,validation_fraction))
            or not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1-train_fraction):
        raise ValueError('split fractions must leave a nonempty test interval')
    assignments = {}
    for instance in instances:
        identifier = instance.instance_id
        score = int(hashlib.sha256(f'{seed}:{identifier}'.encode()).hexdigest()[:16],16)/2**64
        assignments[identifier] = ('train' if score < train_fraction else
                                  'validation' if score < train_fraction+validation_fraction else 'test')
    return assignments
