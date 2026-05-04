#version 330 core
layout(location = 0) in vec2 aPosPx;
layout(location = 1) in vec4 aColorPremul;

uniform vec2 uViewportSizePx;

out vec4 vColor;

void main() {
    vec2 ndc = vec2(
        (aPosPx.x / uViewportSizePx.x) * 2.0 - 1.0,
        1.0 - (aPosPx.y / uViewportSizePx.y) * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor = aColorPremul;
}
