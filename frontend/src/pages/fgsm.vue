<template>
    <div class="page" v-loading="loading" element-loading-background="rgba(0, 0, 0, 0.45)">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" filterable @change="update">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
                </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="params.target = images[params.input]">
                    <el-option v-for="image in Object.keys(images)" :key="image" :value='image'/>
                </el-select>
            </div>
            <div class="item"><div class="title">target</div><el-input class="value" type='number' size="small" v-model="params.target"  @change="update"/></div>
            <div class="item"><div class="title">&epsilon;</div><el-input class="value" size="small" type='number' v-model="params.epsilon"  @change="update"/></div>
            <div class="item"><div class="title">epochs</div><el-input class="value" size="small" type='number' v-model="params.epochs"  @change="update"/></div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="network-inner">
                <div class="input block">
                    <div class="image-wrapper">
                        <el-image class="pixelated" :src="params.input">
                            <div slot="error" class="image-slot">
                                <i class="el-icon-lollipop"></i>
                            </div>
                        </el-image>
                        <div class="caption">Original</div>

                        <el-image class="pixelated saliency" :src="saliency">
                            <div slot="error" class="image-slot">
                                <i class="el-icon-lollipop"></i>
                            </div>
                        </el-image>
                        <div class="predictions">
                            <div class="item" v-for="cls in preds" :key="cls.index">
                                <div class="name">{{cls.class}}</div><div class="value">{{cls.confidence}}</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="add ops">-</div>
                <div class="noise block">
                    <div class="epsilon">{{params.epsilon}}</div>
                    <div class="times">&times;</div>
                    <div class="image-wrapper">
                        <el-image class="pixelated" :src="noise">
                            <div slot="error" class="image-slot">
                                <i class="el-icon-lollipop"></i>
                            </div>
                        </el-image>
                        <div class="caption">Noise</div>
                    </div>
                </div>
                <div class="eq ops">=</div>
                <div class="noised block">
                    <div class="image-wrapper">
                        <el-image class="pixelated" :src="noised">
                            <div slot="error" class="image-slot">
                                <i class="el-icon-lollipop"></i>
                            </div>
                        </el-image>
                        <div class="caption">Noised</div>

                        <el-image class="pixelated saliency" :src="noised_saliency">
                            <div slot="error" class="image-slot">
                                <i class="el-icon-lollipop"></i>
                            </div>
                        </el-image>
                        <div class="predictions">
                            <div class="item" v-for="cls in noised_preds" :key="cls.index">
                                <div class="name">{{cls.class}}</div><div class="value">{{cls.confidence}}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'saliency',
    data() {
        return {
            models: [],
            images: {},
            params: {
                model: 'vgg11',
                input: '',
                target: null,
                epochs: 1,
                epsilon: 0.1
            },
            noise: '',
            noised: '',
            preds: {},
            noised_preds: {},
            saliency: '',
            noised_saliency: '',
            loading: false
        };
    },
    created() {
        this.config()
    },
    sockets: {
        noise(data) {
            this.noise = data
        },
        noised(data) {
            this.noised = data
        },

        saliency(data) {
            this.saliency = data
        },

        noised_saliency(data) {
            this.noised_saliency = data
        },

        preds(data) {
            this.preds = data
        },

        noised_preds(data) {
            this.noised_preds = data
            this.done()
        },

        models(data) {
            this.models = data
        },
        images(data) {
            this.images = data

            this.params.input = Object.keys(data)[0]
            this.params.target = this.images[this.params.input]
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
            this.$socket.emit('get_images')
        },
        update() {
            this.loading = true
            this.noise = null
            this.noised = null
            this.preds = {}
            this.noised_preds = {}
            
            this.$socket.emit("fgsm", this.params);
        },
        done() {
            // if(Object.keys(this.saliency).length && Object.keys(this.guided_saliency).length)
            this.loading = false
        }
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network-inner {
        display: flex;
        flex-flow: row;

        align-items: center;

        .ops {
            flex: 0 0 100px;
            text-align: center;
            font-size: 24px;
        }

        .block {
            flex: 1 1 auto;
        }

        .noise {
            display: flex;
            
            align-items: center;
            // justify-content: center;

            .epsilon {
                flex: 0 0 50px;
                text-align: center;
                font-size: 22px;
            }
            .times {
                flex: 0 0 50px;
                text-align: center;
                font-size: 24px;
            }
        }

        .predictions {
            height: 0;
            .item {
                display: flex;
                flex-flow: column;
                margin: 10px 0;
                align-items: center;
                justify-content: center;

                .name {
                    text-align: center;
                    color: #666;
                }

                .value {
                    color: #333;
                    font-weight: 600;
                }
            }
        }
    }
}
</style>
