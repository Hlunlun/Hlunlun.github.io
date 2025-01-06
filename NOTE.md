# Develop note

## Width of container
- Search using `.has-padding.container`
- Revise in 1486 column of `public/style.css`

## Official Documentation
- [Quick start ](https://gohugo.io/getting-started/quick-start/)

## Math expression
- inline: 包在 <p></p> 裡面，並且用[KATEX](https://katex.org/docs/supported.html#html)的語法
    ```
    <p>\(\x\)</p>
    ```
    就會是 $x$
    
- 如過要自己獨立一個block，就直接包在 `$$` 中
    ```
    $$x$$
    ```