from .Bindings import Next


class NextSerializer:
    def __init__(self):
        next_type_cnt = Next.NumTypes
        self._next_type_to_str = {}
        self._str_to_next_type = {}
        for i in range(next_type_cnt):
            next_type = Next.Type(i)
            name = str(next_type).split('.')[-1]
            self._next_type_to_str[next_type] = name
            self._str_to_next_type[name] = next_type

    def serialize_type(self, next_type: Next.Type) -> str:
        return self._next_type_to_str[next_type]

    def deserialize_type(self, next_type: str) -> Next.Type:
        return self._str_to_next_type[next_type]

    def serialize_next(self, next: Next) -> str:
        return f"{next}"

    def deserialize_next(self, next: str) -> Next:
        next_type, key = next.split('(')
        next_type = self.deserialize_type(next_type)
        key = int(key[:-1])
        return Next(next_type, key)
