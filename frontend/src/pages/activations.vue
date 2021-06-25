<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" @change="update">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="params.target = images[params.input]">
                    <el-option v-for="image in Object.keys(images)" :key="image" :value='image'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">scope</div>
                <el-select class="value" size="small" v-model="params.scope" @change="update">
                    <el-option value='global'/>
                    <el-option value='unit'/>
                    <el-option value='layer'/>
                    <el-option value='channel'/>
                </el-select>
            </div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="unit">
                <div class="layer">
                    <div class="name">input</div>
                    <div class="channels">
                        <div class="channel">
                            <img class="pixelated" :src="params.input"/>
                        </div>
                    </div>
                    <div class="predictions">
                        <div class="item" v-for="cls in predictions" :key="cls.index">
                            <div class="name">{{cls.class}}</div><div class="value">{{cls.confidence}}</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="unit" v-for="(unit, index) in res.units" :key="index">
                <div class="layer" v-for="(layer, name) in unit.layers" :key="name">
                    <div class="name">{{name}}</div>
                    <div class="channels">
                        <div class="channel" v-for="(value, name) in layer.channels" :key="name">
                            <img class="pixelated" :src="value.path" :title="name"/>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            models: [],
            images: {},
            predictions: {},
            res: [],
            params: {
                model: 'vgg11',
                input: '',
                scope: 'layer'
            }
        };
    },
    created() {
        this.config()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        images(data) {
            this.images = data

            this.params.input = Object.keys(data)[0]
        },
        logs(data) {
            console.log(data)
        },
        response_activations(data) {
            // console.log(data)
            this.res = data
            this.$forceUpdate()
        },
        predictions(data) {
            // console.log(data)
            this.predictions = data;
        }
    },
    methods: {
        config() {
            this.$socket.emit("get_models")
            this.$socket.emit("get_images")
        },
        update() {
            this.$socket.emit("activations", this.params);
        },
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        display: flex;

        .predictions {
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

        .unit {
            display: flex;
            padding: 0 5px;

            .layer {
                max-width: 96px;

                display: flex;
                flex-flow: column;
                align-items: center;

                .channels {
                    .channel {
                        width: 96px;
                        // height: 96px;
                        display: flex;
                        align-items: center;
                        justify-content: center;

                        padding: 1px;
                        box-sizing: border-box;
                        img {
                            max-width: 100%;
                            min-width: 16px;
                        }
                    }
                }
            }
        }
    }
}

</style>
