# LLMContext.md

Contexto minimo para retomar sesiones en `src/` sin releer todo el proyecto. Este archivo es el punto de partida para Codex, Claude Code y cualquier otro LLM: leerlo primero y usarlo como mapa inicial antes de explorar codigo.

Existe tambien `../AGENTS.md` en la raiz del proyecto upstream. Leerlo como complemento ya que puede contener informacion adicional o actualizada del proyecto base, pero ante conflictos entre ambos, este archivo (`LLMContext.md`) prevalece por ser especifico del fork del usuario.

## Que es

LichtFeld Studio es una aplicacion nativa C++23/CUDA para 3D Gaussian Splatting. Integra entrenamiento, visualizacion en tiempo real, edicion de escenas/gaussianas, exportacion, plugins Python embebidos y automatizacion via MCP.

Entrada principal: `app/main.cpp`. Parsea CLI (`core/argument_parser.cpp`, header en `core/include/core/argument_parser.hpp`) y enruta a:
- GUI/training: `lfs::app::Application::run` en `app/application.cpp`.
- Conversion de formatos: `app/converter.cpp`.
- Comandos de plugins: `python/runner.*`.
- `--version`, `--help`, warmup.

Target principal CMake: `LichtFeld-Studio` en el `CMakeLists.txt` raiz.

## Preferencias del usuario

- Uso principal: llamadas por CLI. Priorizar rutas CLI, parametros de entrenamiento/conversion/export y comportamiento reproducible por comando.
- Objetivo principal: maxima calidad sin perder demasiado rendimiento. Al proponer defaults o cambios, buscar calidad alta con coste razonable, no el modo mas rapido si degrada demasiado.
- Caso de uso frecuente: escaneos con mascaras para aislar lo escaneado. Las mascaras son importantes para separar sujeto/fondo y mejorar el resultado.
- Datasets habituales: domos de camaras para escanear personas. Cuando el usuario diga "array", asumir salvo contradiccion explicita que habla de un domo de camaras. Tiene un domo de 105 camaras y otro de 48 camaras.
- Cuando un parametro dependa del numero de camaras, preferir expresarlo como proporcion/tanto por 1, no como numero fijo de imagenes/camaras.
- Problemas frecuentes en domos:
  - Falsas transparencias: al ver el sujeto desde multiples angulos, el optimizador puede generar gaussianas semitransparentes en vez de opacas.
  - Recorte insuficiente: picos/artefactos que sobresalen de la silueta deseada, especialmente en zonas concavas (entre piernas, axilas, bajo brazos).
  - Mascaras imperfectas: las mascaras se generan con IA y algunas pueden ser incorrectas. El sistema debe ser robusto a mascaras erroneas aisladas, dejando que la mayoria correcta compense.
- No interesan plugins Python ni UI salvo que el usuario lo pida explicitamente. Evitar gastar tokens en esas zonas para tareas normales de CLI/training/calidad.
- Todo lo que este dentro de `training/smn/` pertenece al fork del usuario, no al proyecto principal upstream.
- Minimizar cambios en archivos del branch original (upstream). Cuando sea posible, colocar la logica nueva en archivos de `training/smn/` y dejar solo one-liners de llamada en los archivos upstream (ej. `trainer.cpp`). Esto reduce conflictos al hacer merge con upstream.
- Comunicacion con el usuario: respuestas y comentarios explicativos en Espanol. Codigo, nombres tecnicos y comentarios dentro del codigo siempre en ingles.
- Evitar quemar tokens en bucles contradictorios o exploraciones circulares. Si hay informacion conflictiva, resumir el conflicto, elegir la hipotesis mas probable con la evidencia disponible y avanzar; pedir aclaracion solo si el riesgo de equivocarse es alto.
- Evitar alucinaciones: no inventar APIs, flags, formatos, rutas ni comportamientos. Verificar en el codigo/docs locales antes de afirmar o implementar.
- Antes de implementar, si hay dudas relevantes sobre intencion, alcance o impacto, preguntar al usuario en vez de asumir.
- Si aparece una solucion que se aleja del codigo actual pero parece claramente superior, explicarla brevemente y preguntar si prefiere ese enfoque antes de aplicarlo.

## Capacidades clave

- Cargar datasets COLMAP, checkpoints y splats/meshes (`PLY`, `SOG`, `SPZ`, `USD/USDA/USDC/USDZ`, formatos mesh via Assimp/OpenMesh).
- Entrenar 3DGS con estrategias `mcmc`, `mrnf` y `igs+`; soporta MCMC/MRNF, GUT, masks, tile mode, sparsity, NGS, bilateral grid y PPISP.
- Visualizar splats/point clouds/meshes/camaras con OpenGL + CUDA interop.
- Editar escena: seleccion, brush/lasso/rect/click, grupos de seleccion, cropbox, ellipsoid, transformaciones, undo/redo.
- Exportar escenas a `PLY`, `SOG`, `SPZ`, `HTML viewer`, `USD*`; exportar video/secuencias.
- Automatizar con MCP y extender con Python plugins/paneles/operadores.

## Arquitectura por carpetas

- `app/`: ejecutable y glue de alto nivel. Registra herramientas MCP especificas de GUI/runtime/operators/sequencer/UI; contiene `main.cpp`, `application.cpp`, `converter.cpp`.
- `core/`: tipos centrales y datos persistentes. `Scene`/`SceneNode`, `SplatData`, `Camera`, parametros, eventos, checkpoints, tensor API propia, CUDA utilities, logging, property system.
- `core/tensor/`: libreria tensor CPU/CUDA sin LibTorch en runtime; expresiones lazy/fusion, broadcasting, reductions, masking, serialization. API publica en `core/include/core/tensor.hpp`.
- `training/`: entrenamiento y optimizacion. `Trainer` coordina dataset, estrategia, optimizer, losses, checkpoint, callbacks Python y estado. Estrategias en `training/strategies/`; rasterizadores/backends en `training/rasterization/`; kernels/losses/components/optimizer en subcarpetas. `training/smn/` es codigo del fork del usuario, no upstream principal.
- `rendering/`: motor/render helpers para viewer: rasterizacion forward, GL resources, framebuffer, shaders, renderers de mesh/grid/bbox/axes/environment/splats.
- `io/`: carga/exportacion de datasets, imagenes, video y formatos (`colmap`, `ply`, `spz`, `sogs`, `usd`, `html`, `transforms`, `nurec_usdz`). Incluye loaders cacheados/pipelined/NV codec.
- `visualizer/`: app interactiva. `VisualizerImpl` une window, GUI, input, scene manager, rendering manager y training manager. Subcarpetas importantes: `gui/`, `rendering/`, `scene/`, `selection/`, `tools/`, `operator/`, `operation/`, `training/`, `sequencer/`.
- `python/`: runtime Python embebido, modulo nanobind `lichtfeld`, stubs `.pyi`, plugins internos `lfs_plugins/`, panels y API para escena/UI/render/selection/io/operators/MCP.
- `mcp/`: servidor/protocolo/registry MCP generico y herramientas compartidas/training context. Las herramientas GUI de la app se registran desde `app/mcp_*.cpp`.
- `sequencer/`: timeline, keyframes, clips, tracks e interpolacion para animacion/camara/video.
- `geometry/`: utilidades geometricas compartidas.

## Datos y modelos principales

- `core::Scene` (`core/include/core/scene.hpp`) es el grafo de escena. Nodos: `SPLAT`, `POINTCLOUD`, `GROUP`, `CROPBOX`, `ELLIPSOID`, `DATASET`, grupos/camaras/imagenes, `MESH`, keyframes.
- `core::SplatData` (`core/include/core/splat_data.hpp`) contiene tensores de gaussianas: means, SH, scaling, rotation, opacity, deleted mask, scene scale. Getters publicos devuelven valores transformados (sigmoid/exp/rot normalizada); accesos `*_raw()` son parametros de optimizacion.
- `training::Trainer` (`training/trainer.hpp`) entrena sobre `Scene`, usa `StrategyFactory`, checkpoints, image loaders, PPISP/bilateral/sparsity y mutex de render.
- `visualizer::SceneManager` es la capa de UI sobre `core::Scene`: carga, seleccion de nodos, contenido activo y operaciones de escena.
- Operaciones visuales usan `visualizer/operator/*` para operators invocables y `visualizer/operation/*` para operaciones con undo/pipeline.

## Flujos tipicos en codigo

- Arranque GUI/training: `main.cpp` -> `parse_args` -> `Application::run` -> crea/configura `VisualizerImpl`.
- Viewer: `VisualizerImpl::run` inicializa ventana SDL, GUI RmlUi/ImGui, input, scene manager, rendering manager, tools y Python bridge.
- Training: `VisualizerImpl::startTraining`/`TrainerManager` -> `training::Trainer::initialize` -> `StrategyFactory::create(params.optimization.strategy)` -> loop `Trainer::train`.
- Render viewport: `visualizer/rendering/RenderingManager` prepara requests/passes; `rendering/` y rasterizer CUDA/GL producen frame.
- Import/export: GUI/MCP/Python llaman a `io/formats/*` o servicios de `visualizer/core/data_loading_service.*`.
- Python: modulo `lichtfeld` se define en `python/lfs/module.cpp`; submodulos importantes: `scene`, `mesh`, `io`, `ui`, `selection`, `rendering`, `animation`, `log`, `app`, `mcp`, `pipeline`, `plugins`, `scripts`. Para este usuario, tratar Python/UI/plugins como secundarios salvo peticion explicita.
- Plugins Python: `python/lfs_plugins/` implementa panels, menus, marketplace, herramientas y compatibilidad; metadata en `pyproject.toml` de cada plugin.

## MCP para agentes

Si la tarea es conducir la app, inspeccionar runtime, manipular GUI/selecciones, entrenar o exportar, usar MCP antes de leer C++.

Config: `.mcp.json` en raiz lanza `scripts/lichtfeld_mcp_bridge.py`; endpoint por defecto `http://localhost:45677/mcp`.

Recursos iniciales recomendados:
- `lichtfeld://runtime/catalog`
- `lichtfeld://runtime/state`
- `lichtfeld://ui/state`
- `lichtfeld://scene/state`
- `lichtfeld://selection/current`

Recursos utiles segun tarea: `lichtfeld://ui/tools`, `lichtfeld://ui/menus`, `lichtfeld://ui/panels`, `lichtfeld://operators/registry`, `lichtfeld://operators/modal_state`, `lichtfeld://scene/nodes`, `lichtfeld://scene/selected_nodes`, `lichtfeld://runtime/jobs/<job_id>`, `lichtfeld://runtime/events/<event_type>`.

Job ids conocidos: `training.main`, `editor.python`, `import.dataset`, `export.scene`, `export.video`, `operator.modal`.

Regla: no adivinar ids de operators/menu/panel/tool; leer registry/recurso primero. Tras escribir seleccion, confirmar con `selection.get` o `lichtfeld://selection/current`.

## Build, tests y dependencias

- CMake raiz requiere CUDA, C++23, vcpkg, SDL3, OpenGL, OpenImageIO, spdlog, nanobind/Python, etc. CUDA default: fatbin; opciones relevantes `BUILD_PORTABLE`, `BUILD_CUDA_PTX_ONLY`, `BUILD_CUDA_FATBIN`, `BUILD_TESTS`, `ENABLE_CUDA_GL_INTEROP`.
- Modulos CMake principales: `lfs_core`, `lfs_training`, `lfs_io`, `lfs_rendering`, `lfs_sequencer`, `lfs_visualizer`, `lfs_geometry`, `lfs_mcp`, `lfs_python_runtime`, `lfs_py`.
- Build Windows existente suele estar en `../BUILD_NoTorch` o `../BUILD`. Comando orientativo desde `src/`: `cmake --build ..\BUILD_NoTorch --config Release --target LichtFeld-Studio`.
- Tests en `../tests`; CMake tiene `BUILD_TESTS`. Hay tests de tensor, CUDA, rasterizers, training strategies, formats, MCP, Python, UI/operator/selection/sequencer.
- Si `git status` falla por "dubious ownership", no modificar configuracion global salvo que la tarea necesite Git y el usuario lo autorice.

## Convenciones practicas para editar

- Mantener cambios acotados al modulo afectado; no refactorizar de paso.
- Evitar bucles de lectura sin salida: leer primero el punto de entrada y 2-4 archivos directamente relacionados; ampliar solo si aparece una dependencia concreta.
- No crear patrones nuevos sin necesidad: seguir la arquitectura actual salvo que el usuario apruebe explicitamente una alternativa superior.
- Respetar namespaces: `lfs::app`, `lfs::core`, `lfs::training`, `lfs::io`, `lfs::rendering`, `lfs::vis`, `lfs::mcp`, `lfs::python`.
- Headers publicos de `core` estan en `core/include/core/*.hpp`; implementaciones en `core/*.cpp`. Buscar APIs en el header, no en el `.cpp`.
- Para datos GPU/tensores, preferir `core::Tensor` y utilidades existentes; no introducir LibTorch como dependencia runtime.
- Para GUI, comprobar si el panel es nativo C++ (`visualizer/gui/*`) o Python/RmlUi (`python/lfs_plugins/*`, `visualizer/gui/rmlui/resources/*`) antes de editar, pero no entrar en UI/plugins salvo peticion explicita.
- Para operators/herramientas, registrar en `visualizer/operator/` o `visualizer/tools/` y exponer por MCP/Python solo si hace falta.
- Para formatos, mirar `io/formats/*` y tests existentes antes de tocar exporters/importers.
- Para entrenamiento, preservar compatibilidad de checkpoints y nombres canonicos de estrategia (`mcmc`, `mrnf`, `igs+`).
- Para recomendaciones de training CLI, priorizar calidad alta con rendimiento razonable, mascaras para aislar el escaneo y parametros proporcionales al numero de camaras cuando aplique.
- Para operaciones largas desde MCP, usar recursos/jobs/events en vez de sleeps.

## Archivos de arranque rapido

- Producto/readme: `../README.md`
- CMake raiz: `../CMakeLists.txt`
- CLI: `core/argument_parser.cpp`, `core/include/core/argument_parser.hpp`
- Main: `app/main.cpp`
- Application: `app/application.cpp`
- Viewer central: `visualizer/visualizer_impl.*`
- Escena: `core/include/core/scene.hpp`, `core/scene.cpp`
- Splat data: `core/include/core/splat_data.hpp`, `core/splat_data.cpp`
- Tensor API: `core/include/core/tensor.hpp`, `core/tensor/`
- Trainer: `training/trainer.*`
- Estrategias: `training/strategies/`
- IO formatos: `io/formats/`
- Python module: `python/lfs/module.cpp`
- Plugins Python: `python/lfs_plugins/`
- MCP core: `mcp/*`
- MCP app/GUI: `app/mcp_*.cpp`
- Docs MCP: `../docs/docs/development/mcp/`
