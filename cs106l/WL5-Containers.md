Supplemental Material
# When to use which sequence container?
| 🍁 | [std::vector](https://en.cppreference.com/w/cpp/container/vector) | [std::deque](https://en.cppreference.com/w/cpp/container/deque) | [std::list](https://en.cppreference.com/w/cpp/container/list) |
| ------------------- | ------------------ | -------------- | -------------- |
| Indexed Access      | **Super Fast**     | **Fast**       | Impossible     |
| Insert/remove front | Slow               | **Fast**       | **Fast**       |
| Insert/remove back  | **Super Fast**     | **Very Fast**  | **Fast**       |
| Ins/rem elsewhere   | Slow               | **Fast**       | **Very Fast**  |
| Memory              | **Low**            | High           | High           |
| Splicing/Joining    | Slow               | Very Slow      | **Fast**       |
| Stability <br>Iterators, concurrency | Poor | Very Poor   | **Good**       |

- vector: use for most purposes
- deque: frequent insert/remove at front
- list: very rarely - if need splitting/joining

Containers:
1. [std::stack](https://en.cppreference.com/w/cpp/container/stack)
2. [std::priority_queue](https://en.cppreference.com/w/cpp/container/priority_queue)

Associative Containers:
1. [std::set](https://en.cppreference.com/w/cpp/container/set)
2. [std:map](https://en.cppreference.com/w/cpp/container/map)  
