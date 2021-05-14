<template>
    <z-page :menu="menu" />
</template>

<script>
    import zpage from "./zpage";

    import 'vue-awesome/icons/fire'

    export default {
        data() {
            return {
                menu: [
                    {
                        icon: 'fire',
                        index: '/activations',
                        title: 'Activations'
                    },
                    {
                        icon: 'fire',
                        index: '/filters',
                        title: 'Filters'
                    },
                    {
                        icon: 'fire',
                        index: '/class_max',
                        title: 'Class Maximum'
                    },
                    {
                        icon: 'fire',
                        index: '/activation_max',
                        title: 'Activation Maximum'
                    },                    
                    {
                        icon: 'fire',
                        index: '/deep_dream',
                        title: 'Deep Dream'
                    },
                    {
                        icon: 'fire',
                        index: '/saliency',
                        title: 'Saliency'
                    },
                    {
                        icon: 'fire',
                        index: '/smooth_grad',
                        title: 'SmoothGrad'
                    },
                    {
                        icon: 'fire',
                        index: '/gradcam',
                        title: 'GradCam'
                    }
                ],
            }
        },
        components: {
            'z-page' : zpage
        }
    }
</script>

<style rel="stylesheet/scss" lang="scss">
.page {
    display: flex;
    flex-flow: column;

    .menu {
        flex: 0 0 auto;

        background-color: white;
        padding: 10px;
        box-sizing: border-box;

        display: flex;
        flex-flow: row wrap;
        
        .item {
            flex: 0 0 auto;
            
            width: 250px;
            display: flex;
            align-items: center;
            flex-flow: row;
            height: 50px;

            .title {
                width: 75px;
                text-align: right;
                padding: 0 10px;
                box-sizing: border-box;
            }

            .value {
                width: 175px;
            }

            .button {
                margin-left: 50px;
            }
        }
    }
    
    .network {
        overflow-y: auto;
        flex: 1 1 auto;
        padding: 25px;
    }
}

.image-wrapper {
    display: flex;
    flex-flow: column;

    .image {
        flex: 0 0 auto;
    }
    .caption {
        flex: 0 0 30px;
        padding-top: 5px;
        text-align: center;
    }
}

</style>