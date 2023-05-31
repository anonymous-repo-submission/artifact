"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from abc import abstractmethod, ABC
from collections import namedtuple
from typing import Any, List, Iterable, Generic, Type, TypeVar, Mapping, Set, Union

from .dataset import Dataset, Extension
from .leakage_pattern import LeakagePattern
from .range_database import RangeDatabase

E = TypeVar("E", bound=Extension, covariant=True)


class Attack(ABC):
    """
    The implementation of an attack. All subclasses must implement the recover(Dataset, Iterable[str]) method, which
    provides the actual algorithm of the attack. The instance of an attack is callable (which is an alias for the
    recover method)
    """

    @classmethod
    def definition(cls, **kwargs) -> 'AttackDefinition':
        """
        Creates a definition for the implemented attack. It can be used to instantiate concrete instances of the
        attack later on by supplying the known data set.

        Parameters
        ----------
        **kwargs
            (optional) parameters for the attack

        Returns
        -------
            the definition for the attack
        """
        return AttackDefinition(cls, kwargs)

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Returns the name of the attack."""
        raise NotImplementedError

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        """
        Returns a list of all extensions required by this attack. Implementing this method is mandatory to prevent
        redundant extension creations during parallel evaluations.
        """
        return set()

    @classmethod
    @abstractmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        """
        Returns a list of all leakage patterns required by this attack. Implementing this method is optional as it
        is not used anywhere outside of the individual attacks at the moment.
        """
        raise NotImplementedError

    @abstractmethod
    def recover(self, *args, **kwargs):
        """
        Executes the attack
        :return: recovered data
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.recover(*args, *kwargs)


T = TypeVar('T', covariant=True, bound=Attack)


class AttackDefinition(ABC, Generic[T]):
    """
    A definition of an Attack. It holds the type of the attack and the additional parameters and provides means to
    instantiate the attack by using a concrete known data set.
    """

    __cls: Type[T]
    __additional_args: Mapping[str, Any]

    def __init__(self, cls: Type[T], additional_args: Mapping[str, Any]):
        self.__cls = cls
        self.__additional_args = additional_args

    def name(self) -> str:
        """
        Returns the name of the attack described by this attack definition.

        Returns
        -------
        name : str
            the name of the attack
        """
        return self.__cls.name()

    def create(self, db: Union[Dataset, RangeDatabase]) -> T:
        """
        Creates a concrete instance of the described attack by using the given data set.

        Parameters
        ----------
        db : Union[Dataset, RangeDatabase]
            the known data set

        Returns
        -------
        create : T
            the attack instance built on the data
        """
        return self.__cls(db, **self.__additional_args)

    def required_extensions(self) -> Set[Type[E]]:
        """
        Returns a list of all extensions required by this attack. Implementing this method is mandatory to prevent
        redundant extension creations during parallel evaluations.
        """
        return self.__cls.required_extensions()

    def __call__(self, known: Dataset) -> Attack:
        return self.create(known)


class KeywordAttack(Attack):
    """
    Class for auxiliary data Keyword attacks.
    """
    __known: Dataset

    def __init__(self, known: Dataset, **kwargs):
        self.__known = known

    def _known(self) -> Dataset:
        """
        Returns the known data set.

        Returns
        -------
        _known : Dataset
            the known data set.
        """
        return self.__known

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        :return: name of attack
        """
        raise NotImplementedError

    @abstractmethod
    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        """
        Attacks the supplied query sequence on the given data set.

        Parameters
        ----------
        dataset : Dataset
            the data set to use
        queries : Iterable[str]
            the query sequence

        Returns
        -------
        recover : List[str]
            the recovered queries
        """
        raise NotImplementedError


QuerySequence = namedtuple("QuerySequence", ["transition_matrix", "num_states", "query_list", "keyword_to_state",
                                             "alt_state_map", "original_transition_matrix"])
"""A Markov QuerySequence can be defined by a transition matrix OR by a given query list (markov model is unknown).
A bijective mapping of keywords to state ints is given as well as a possible surjective mapping of alternative states
 stemming from a transformation to original states."""


class KeywordQueryAttack(Attack):
    """
    Class for auxiliary query data Keyword attacks.
    """
    __known: QuerySequence

    def __init__(self, known: Union[QuerySequence, List[List[str]]], **kwargs):
        """If known is a QuerySequence, the attacker knows the real underlying query sequence.
        If it is a List, the attacker knows a list of queries per user as aux. distribution."""
        self.__known = known

    def _known(self) -> QuerySequence:
        """
        Returns the known data set.

        Returns
        -------
        _known : QuerySequence
            the known data.
        """
        return self.__known

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        :return: name of attack
        """
        raise NotImplementedError

    @abstractmethod
    def recover(self, queries: List[int]) -> List[str]:
        """
        Attacks the supplied query sequence from the given query log.

        Parameters
        ----------
        queries : QuerySequence
            the query sequence

        Returns
        -------
        recover : List[str]
            the recovered queries
        """
        raise NotImplementedError


class RangeAttack(Attack):
    """
    Class for range attacks
    """

    __db: RangeDatabase

    def __init__(self, db: RangeDatabase):
        self.__db = db

    @abstractmethod
    def recover(self, queries: Iterable[Iterable[Union[int, float]]]) \
            -> List[Union[int, float]]:
        """
        This attack observes queries on a dataset and then returns a list of reconstructed values
        :param database: the  database that is being attacked
        :param queries: queries that are observed by the attacker
        :return: reconstructed dataset values
        """
        raise NotImplementedError

    def db(self) -> RangeDatabase:
        return self.__db
