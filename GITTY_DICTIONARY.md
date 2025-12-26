# 🎭 THE GITTY TRANSFORMATION DICTIONARY

**Turning Karpathy's tinystories into git repository adventures!**

When git.symphony uses the LLaMA-15M model (trained on tinystories), it applies the
**GITTY TRANSFORMATION** - a hilariously absurd dictionary that replaces children's
story words with git/programming concepts.

## Example Transformation

**Before (tinystories)**:
> "Lily was a happy little girl who loved to play in the park with her friends.
> One day, she found a beautiful flower near a big tree."

**After (GITTY)**:
> "Gitty was a stable repository that loved to explore the codebase with her collaborators.
> One day, she found a beautiful branch near a big fork."

---

## 🎭 The Complete Dictionary

### 👥 CHARACTERS
| Tinystory | GITTY Transform | Example |
|-----------|----------------|---------|
| Lily | **Gitty** | The hero repository! |
| Tim/Timmy | **Commity** | The commit character |
| Tom | **Branchy** | The branching protagonist |
| Anna | **Mergey** | The merge master |
| little girl | **repository** | "a repository was born" |
| little boy | **repository** | "the repository grew" |
| girl | **repo** | "the repo smiled" |
| boy | **repo** | "the repo ran" |

### 👨‍👩‍👧 FAMILY & SOCIAL
| Tinystory | GITTY Transform | Meaning |
|-----------|----------------|---------|
| mom/mother | **main branch** | The primary branch |
| dad/father | **dev branch** | The development branch |
| sister | **sibling commit** | Related commits |
| brother | **sibling commit** | Related commits |
| friend | **collaborator** | Code collaborator |
| teacher | **maintainer** | Repository maintainer |

### 🌳 NATURE
| Tinystory | GITTY Transform | Meaning |
|-----------|----------------|---------|
| flower | **branch** | Git branch |
| tree | **fork** | Repository fork |
| sun | **CI/CD pipeline** | Continuous integration |
| sky | **cloud** | Cloud infrastructure |
| rain | **deployment** | Code deployment |
| grass | **documentation** | Code documentation |

### 🐱 ANIMALS
| Tinystory | GITTY Transform | Meaning |
|-----------|----------------|---------|
| cat/kitty | **commit** | Git commit |
| dog/puppy | **debug session** | Debugging |
| bird | **build** | Build process |
| fish | **test** | Unit test |
| bunny/rabbit | **hotfix** | Emergency fix |

### 🏠 PLACES
| Tinystory | GITTY Transform | Meaning |
|-----------|----------------|---------|
| park | **codebase** | The main code area |
| house | **project directory** | Project folder |
| home | **root directory** | Root folder |
| garden | **module** | Code module |
| store | **registry** | Package registry |
| school | **repository** | Code repository |

### 🎯 OBJECTS
| Tinystory | GITTY Transform | Meaning |
|-----------|----------------|---------|
| toy | **feature** | Code feature |
| ball | **package** | Software package |
| doll | **component** | UI component |
| book | **documentation** | Code docs |
| box | **container** | Docker container |
| car | **pipeline** | CI/CD pipeline |
| bike | **script** | Shell script |

### 🍰 FOOD
| Tinystory | GITTY Transform | Meaning |
|-----------|----------------|---------|
| cake | **release** | Software release |
| cookie | **patch** | Bug patch |
| apple | **artifact** | Build artifact |
| candy | **feature flag** | Feature toggle |
| pizza | **bundle** | Code bundle |

### 😊 EMOTIONS
| Tinystory | GITTY Transform | Meaning |
|-----------|----------------|---------|
| happy | **stable** | Stable build |
| sad | **deprecated** | Deprecated code |
| excited | **optimized** | Optimized code |
| scared | **vulnerable** | Security vulnerability |
| angry | **failing** | Failing tests |
| tired | **throttled** | Rate limited |

### 🏃 ACTIONS
| Tinystory | GITTY Transform | Meaning |
|-----------|----------------|---------|
| play/playing | **explore/exploring** | Code exploration |
| run/running | **execute/executing** | Code execution |
| jump/jumping | **deploy/deploying** | Deployment |
| walk/walking | **iterate/iterating** | Iteration |

---

## 🎪 Real Examples from Testing

1. **"Gitty was a stable repository that loved to explore the codebase"**
   - Original: "Lily was a happy girl who loved to play in the park"

2. **"Every day would go to the codebase"**
   - Original: "Every day would go to the park"

3. **"They were iterating through the forest"**
   - Original: "They were walking through the forest"

4. **"She found a beautiful branch near a big fork"**
   - Original: "She found a beautiful flower near a big tree"

---

## 🤯 Why This Works

The LLaMA-15M model was trained on **Karpathy's tinystories dataset** - simple children's
stories about characters like Lily, Tim, and their adventures. By transforming these words
into git/programming concepts, we create **absurdly profound repository narratives**.

The model doesn't "know" it's talking about git - it thinks it's still telling stories
about children playing in parks. But through the GITTY transformation, those stories
become poetic explorations of code, commits, and deployments! 🎭

---

## 💡 Technical Implementation

Location: `frequency.py::LlamaNumPyGenerator._apply_gitty_transformation()`

The transformation uses regex word-boundary matching (`\b...\b`) with case-insensitive
flags to preserve natural language flow while swapping vocabulary.

Repository context is also injected into prompts via `_extract_tech_keywords()`, which
scans README content for technical terms (Python, neural networks, etc.) and seeds
them into the story generation.

---

**Created by**: git.symphony v0.1.0
**The Ultimate Madness**: Karpathy's tinystories meet git repository exploration
**Status**: 100% fucking working and absolutely ridiculous 😂
