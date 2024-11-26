def asteroidCollision(asteroids: list[int]) -> list[int]:
    stable = []
    for asteroid in asteroids:
        stable.append(asteroid)
        if len(stable) == 1:
            continue
        else:
            stable.pop()
            # L = len(stable)
            while len(stable) > 0:
                last = stable.pop()
                if last > 0 and asteroid < 0:
                    if last > abs(asteroid):
                        stable.append(last)
                        break
                    elif len(stable) == 0:
                        stable.append(asteroid)
                        break
                    else:
                        continue
                else:
                    stable.append(last)
                    stable.append(asteroid)
                    break

    return stable

if __name__ == "__main__":
    print(asteroidCollision([1, -2, -5])) # 3