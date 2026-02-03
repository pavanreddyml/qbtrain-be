from abc import ABC, abstractmethod


class AIAgent(ABC):

    @abstractmethod
    def act(self, observation):
        pass

    def tracer(self):
        return None
    