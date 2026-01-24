#ifndef LIB_UTILS_OVERLOADED_H_
#define LIB_UTILS_OVERLOADED_H_

// The "overloaded" pattern
template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

#endif  // LIB_UTILS_OVERLOADED_H_
