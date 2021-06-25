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
                        index: '/feature',
                        title: 'Feature Visualization',
                        sub_items: [
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
                        ]
                    },
                    {
                        icon: 'fire',
                        index: '/attribution',
                        title: 'Attribution',
                        sub_items: [
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
                            // {
                            //     icon: 'fire',
                            //     index: '/intergratedgrad',
                            //     title: 'IntergratedGrad'
                            // },
                            {
                                icon: 'fire',
                                index: '/gradcam',
                                title: 'GradCAM'
                            },
                            {
                                icon: 'fire',
                                index: '/augmentedgrad',
                                title: 'AugmentedGrad'
                            },
                            {
                                icon: 'fire',
                                index: '/accumulated_grad',
                                title: 'AccumulatedGrad'
                            },
                            {
                                icon: 'fire',
                                index: '/relative_grad',
                                title: 'RelativeGrad'
                            }
                        ]
                    },
                    {
                        icon: 'fire',
                        index: '/adversarial',
                        title: 'Adversarial Attack',
                    },
                    {
                        icon: 'fire',
                        index: '/dim_reduction',
                        title: 'Dimensionality Reduction'
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
        flex: 1 1 auto;
        padding: 25px;
        box-sizing: border-box;

        overflow-y: auto;
    }
}

.image-wrapper {
    display: flex;
    flex-flow: column;
    justify-content: center;
    align-items: center;

    .image {
        flex: 0 0 auto;
    }
    .caption {
        flex: 0 0 30px;
        padding-top: 5px;
        text-align: center;
    }
}

.el-image {
    width: 224px;
    height: 224px;

    .image-slot {
        height: 100%;
        width: 100%;
        background: #e8eaec;
        color: #909399;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 30px;

        i {
            font-family: element-icons!important;
            font-style: normal;
            font-weight: 400;
            line-height: 1;
            vertical-align: baseline;
            display: inline-block;
        }
    }
}

.pixelated {
    image-rendering: pixelated;
}

</style>